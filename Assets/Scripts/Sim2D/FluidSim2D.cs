using Seb.Fluid2D.Rendering;
using Seb.Helpers;
using UnityEngine;
using Unity.Mathematics;
using UnityEngine.Serialization;

namespace Seb.Fluid2D.Simulation
{
	public class FluidSim2D : MonoBehaviour
	{
		[System.Serializable]
		public struct Obstacle2D
		{
			public Vector2 centre;
			public Vector2 size;
			[Tooltip("Rotation in degrees (counter-clockwise)")]
			public float rotation;
		}

		[System.Serializable]
		public struct PolygonObstacle2D
		{
			[Tooltip("Convex polygon vertices in world-space, listed in clockwise or counter-clockwise order.")]
			public Vector2[] vertices;
		}

		public event System.Action SimulationStepCompleted;

		[Header("Simulation Settings")]
		public float timeScale = 1;
		public float maxTimestepFPS = 60; // if time-step dips lower than this fps, simulation will run slower (set to 0 to disable)
		public int iterationsPerFrame;
		public float gravity;
		[Range(0, 1)] public float collisionDamping = 0.95f;
		public float smoothingRadius = 2;
		public float targetDensity;
		public float pressureMultiplier;
		public float nearPressureMultiplier;
		public float viscosityStrength;
		public Vector2 boundsSize;

		[Header("Obstacles")]
		[Tooltip("List of rectangular obstacles (centre + size) the fluid should collide with.")]
		public Obstacle2D[] obstacles;

		[Tooltip("Optional list of convex polygon obstacles (uses vertices in world-space).")]
		public PolygonObstacle2D[] polygonObstacles;

		// Legacy single-obstacle fields (kept for backwards compatibility with existing scenes)
		[FormerlySerializedAs("obstacleSize")]
		[HideInInspector] public Vector2 legacyObstacleSize;
		[FormerlySerializedAs("obstacleCentre")]
		[HideInInspector] public Vector2 legacyObstacleCentre;

		[Header("Interaction Settings")]
		public float interactionRadius;

		public float interactionStrength;

		[Header("References")]
		public ComputeShader compute;

		public Spawner2D spawner2D;

		// Buffers
		public ComputeBuffer positionBuffer { get; private set; }
		public ComputeBuffer velocityBuffer { get; private set; }
		public ComputeBuffer densityBuffer { get; private set; }

		ComputeBuffer sortTarget_Position;
		ComputeBuffer sortTarget_PredicitedPosition;
		ComputeBuffer sortTarget_Velocity;

		ComputeBuffer predictedPositionBuffer;
		SpatialHash spatialHash;
		ComputeBuffer obstacleBuffer;
		ComputeBuffer obstacleRotationBuffer;

    	// Polygon obstacles
		const int MaxPolygonVertices = 8;
		ComputeBuffer polygonVertexBuffer;      // float2 [numPolygons * MaxPolygonVertices]
		ComputeBuffer polygonVertexCountBuffer; // int [numPolygons]

		// Kernel IDs
		const int externalForcesKernel = 0;
		const int spatialHashKernel = 1;
		const int reorderKernel = 2;
		const int copybackKernel = 3;
		const int densityKernel = 4;
		const int pressureKernel = 5;
		const int viscosityKernel = 6;
		const int updatePositionKernel = 7;

		// State
		bool isPaused;
		Spawner2D.ParticleSpawnData spawnData;
		bool pauseNextFrame;

		public int numParticles { get; private set; }


		void Start()
		{
			Debug.Log("Controls: Space = Play/Pause, R = Reset, LMB = Attract, RMB = Repel");

			Init();
		}

		void Init()
		{
			float deltaTime = 1 / 60f;
			Time.fixedDeltaTime = deltaTime;

			spawnData = spawner2D.GetSpawnData();
			numParticles = spawnData.positions.Length;
			spatialHash = new SpatialHash(numParticles);

			// Create buffers
			positionBuffer = ComputeHelper.CreateStructuredBuffer<float2>(numParticles);
			predictedPositionBuffer = ComputeHelper.CreateStructuredBuffer<float2>(numParticles);
			velocityBuffer = ComputeHelper.CreateStructuredBuffer<float2>(numParticles);
			densityBuffer = ComputeHelper.CreateStructuredBuffer<float2>(numParticles);

			sortTarget_Position = ComputeHelper.CreateStructuredBuffer<float2>(numParticles);
			sortTarget_PredicitedPosition = ComputeHelper.CreateStructuredBuffer<float2>(numParticles);
			sortTarget_Velocity = ComputeHelper.CreateStructuredBuffer<float2>(numParticles);

			// Set buffer data
			SetInitialBufferData(spawnData);

			// Init compute
			ComputeHelper.SetBuffer(compute, positionBuffer, "Positions", externalForcesKernel, updatePositionKernel, reorderKernel, copybackKernel);
			ComputeHelper.SetBuffer(compute, predictedPositionBuffer, "PredictedPositions", externalForcesKernel, spatialHashKernel, densityKernel, pressureKernel, viscosityKernel, reorderKernel, copybackKernel);
			ComputeHelper.SetBuffer(compute, velocityBuffer, "Velocities", externalForcesKernel, pressureKernel, viscosityKernel, updatePositionKernel, reorderKernel, copybackKernel);
			ComputeHelper.SetBuffer(compute, densityBuffer, "Densities", densityKernel, pressureKernel, viscosityKernel);

			ComputeHelper.SetBuffer(compute, spatialHash.SpatialIndices, "SortedIndices", spatialHashKernel, reorderKernel);
			ComputeHelper.SetBuffer(compute, spatialHash.SpatialOffsets, "SpatialOffsets", spatialHashKernel, densityKernel, pressureKernel, viscosityKernel);
			ComputeHelper.SetBuffer(compute, spatialHash.SpatialKeys, "SpatialKeys", spatialHashKernel, densityKernel, pressureKernel, viscosityKernel);

			ComputeHelper.SetBuffer(compute, sortTarget_Position, "SortTarget_Positions", reorderKernel, copybackKernel);
			ComputeHelper.SetBuffer(compute, sortTarget_PredicitedPosition, "SortTarget_PredictedPositions", reorderKernel, copybackKernel);
			ComputeHelper.SetBuffer(compute, sortTarget_Velocity, "SortTarget_Velocities", reorderKernel, copybackKernel);

			compute.SetInt("numParticles", numParticles);

			InitObstacles();
			InitPolygonObstacles();
		}


		void Update()
		{
			if (!isPaused)
			{
				float maxDeltaTime = maxTimestepFPS > 0 ? 1 / maxTimestepFPS : float.PositiveInfinity; // If framerate dips too low, run the simulation slower than real-time
				float dt = Mathf.Min(Time.deltaTime * timeScale, maxDeltaTime);
				RunSimulationFrame(dt);
			}

			if (pauseNextFrame)
			{
				isPaused = true;
				pauseNextFrame = false;
			}

			HandleInput();
		}

		void RunSimulationFrame(float frameTime)
		{
			float timeStep = frameTime / iterationsPerFrame;

			UpdateSettings(timeStep);

			for (int i = 0; i < iterationsPerFrame; i++)
			{
				RunSimulationStep();
				SimulationStepCompleted?.Invoke();
			}
		}

		void RunSimulationStep()
		{
			ComputeHelper.Dispatch(compute, numParticles, kernelIndex: externalForcesKernel);

			RunSpatial();

			ComputeHelper.Dispatch(compute, numParticles, kernelIndex: densityKernel);
			ComputeHelper.Dispatch(compute, numParticles, kernelIndex: pressureKernel);
			ComputeHelper.Dispatch(compute, numParticles, kernelIndex: viscosityKernel);
			ComputeHelper.Dispatch(compute, numParticles, kernelIndex: updatePositionKernel);
		}

		void RunSpatial()
		{
			ComputeHelper.Dispatch(compute, numParticles, kernelIndex: spatialHashKernel);
			spatialHash.Run();

			ComputeHelper.Dispatch(compute, numParticles, kernelIndex: reorderKernel);
			ComputeHelper.Dispatch(compute, numParticles, kernelIndex: copybackKernel);
		}

		void UpdateSettings(float deltaTime)
		{
			compute.SetFloat("deltaTime", deltaTime);
			compute.SetFloat("gravity", gravity);
			compute.SetFloat("collisionDamping", collisionDamping);
			compute.SetFloat("smoothingRadius", smoothingRadius);
			compute.SetFloat("targetDensity", targetDensity);
			compute.SetFloat("pressureMultiplier", pressureMultiplier);
			compute.SetFloat("nearPressureMultiplier", nearPressureMultiplier);
			compute.SetFloat("viscosityStrength", viscosityStrength);
			compute.SetVector("boundsSize", boundsSize);

			// Obstacles are uploaded once in InitObstacles. Here we just ensure count is up to date
			int obstacleCount = obstacles != null ? obstacles.Length : 0;
			compute.SetInt("obstacleCount", obstacleCount);

			int polygonCount = polygonObstacles != null ? polygonObstacles.Length : 0;
			compute.SetInt("polygonObstacleCount", polygonCount);
			compute.SetInt("polygonMaxVertices", MaxPolygonVertices);

			compute.SetFloat("Poly6ScalingFactor", 4 / (Mathf.PI * Mathf.Pow(smoothingRadius, 8)));
			compute.SetFloat("SpikyPow3ScalingFactor", 10 / (Mathf.PI * Mathf.Pow(smoothingRadius, 5)));
			compute.SetFloat("SpikyPow2ScalingFactor", 6 / (Mathf.PI * Mathf.Pow(smoothingRadius, 4)));
			compute.SetFloat("SpikyPow3DerivativeScalingFactor", 30 / (Mathf.Pow(smoothingRadius, 5) * Mathf.PI));
			compute.SetFloat("SpikyPow2DerivativeScalingFactor", 12 / (Mathf.Pow(smoothingRadius, 4) * Mathf.PI));

			// Mouse interaction settings:
			Vector2 mousePos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
			bool isPullInteraction = Input.GetMouseButton(0);
			bool isPushInteraction = Input.GetMouseButton(1);
			float currInteractStrength = 0;
			if (isPushInteraction || isPullInteraction)
			{
				currInteractStrength = isPushInteraction ? -interactionStrength : interactionStrength;
			}

			compute.SetVector("interactionInputPoint", mousePos);
			compute.SetFloat("interactionInputStrength", currInteractStrength);
			compute.SetFloat("interactionInputRadius", interactionRadius);
		}

		void SetInitialBufferData(Spawner2D.ParticleSpawnData spawnData)
		{
			float2[] allPoints = new float2[spawnData.positions.Length]; //
			System.Array.Copy(spawnData.positions, allPoints, spawnData.positions.Length);

			positionBuffer.SetData(allPoints);
			predictedPositionBuffer.SetData(allPoints);
			velocityBuffer.SetData(spawnData.velocities);
		}

		void HandleInput()
		{
			if (Input.GetKeyDown(KeyCode.Space))
			{
				isPaused = !isPaused;
			}

			if (Input.GetKeyDown(KeyCode.RightArrow))
			{
				isPaused = false;
				pauseNextFrame = true;
			}

			if (Input.GetKeyDown(KeyCode.R))
			{
				isPaused = true;
				// Reset positions, the run single frame to get density etc (for debug purposes) and then reset positions again
				SetInitialBufferData(spawnData);
				RunSimulationStep();
				SetInitialBufferData(spawnData);
			}
		}


		void OnDestroy()
		{
			ComputeHelper.Release(positionBuffer, predictedPositionBuffer, velocityBuffer, densityBuffer, sortTarget_Position, sortTarget_Velocity, sortTarget_PredicitedPosition, obstacleBuffer, obstacleRotationBuffer, polygonVertexBuffer, polygonVertexCountBuffer);
			spatialHash.Release();
		}


		void OnDrawGizmos()
		{
			Gizmos.color = new Color(0, 1, 0, 0.4f);
			Gizmos.DrawWireCube(Vector2.zero, boundsSize);

			// Draw obstacles
			Matrix4x4 oldMatrix = Gizmos.matrix;
			if (obstacles != null && obstacles.Length > 0)
			{
				foreach (var obstacle in obstacles)
				{
					Gizmos.matrix = Matrix4x4.TRS(obstacle.centre, Quaternion.Euler(0, 0, obstacle.rotation), Vector3.one);
					Gizmos.DrawWireCube(Vector3.zero, obstacle.size);
				}
			}
			else if (legacyObstacleSize != Vector2.zero)
			{
				// Fallback for scenes that still use the old single-obstacle fields
				Gizmos.matrix = Matrix4x4.TRS(legacyObstacleCentre, Quaternion.identity, Vector3.one);
				Gizmos.DrawWireCube(Vector3.zero, legacyObstacleSize);
			}
			Gizmos.matrix = oldMatrix;

			// Draw polygon obstacles
			if (polygonObstacles != null)
			{
				Gizmos.color = new Color(1, 0.5f, 0, 0.6f);
				foreach (var poly in polygonObstacles)
				{
					if (poly.vertices == null || poly.vertices.Length < 2) continue;
					for (int i = 0; i < poly.vertices.Length; i++)
					{
						Vector3 a = poly.vertices[i];
						Vector3 b = poly.vertices[(i + 1) % poly.vertices.Length];
						Gizmos.DrawLine(a, b);
					}
				}
			}

			if (Application.isPlaying)
			{
				Vector2 mousePos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
				bool isPullInteraction = Input.GetMouseButton(0);
				bool isPushInteraction = Input.GetMouseButton(1);
				bool isInteracting = isPullInteraction || isPushInteraction;
				if (isInteracting)
				{
					Gizmos.color = isPullInteraction ? Color.green : Color.red;
					Gizmos.DrawWireSphere(mousePos, interactionRadius);
				}
			}
		}

		// --- Obstacles ---

		void InitObstacles()
		{
			// If no obstacles have been explicitly set up, but legacy values exist,
			// convert the legacy single obstacle into the new obstacle array.
			if ((obstacles == null || obstacles.Length == 0) && legacyObstacleSize != Vector2.zero)
			{
				obstacles = new Obstacle2D[1];
				obstacles[0] = new Obstacle2D
				{
					centre = legacyObstacleCentre,
					size = legacyObstacleSize
				};
			}

			int obstacleCount = obstacles != null ? obstacles.Length : 0;

			// Release any previous buffers
			ComputeHelper.Release(obstacleBuffer, obstacleRotationBuffer);

			if (obstacleCount == 0)
			{
				compute.SetInt("obstacleCount", 0);
				return;
			}

			// Pack obstacle data:
			// - obstacleData: (centre.xy, halfSize.xy)
			// - rotationData: (cos(theta), sin(theta))
			float4[] obstacleData = new float4[obstacleCount];
			float2[] rotationData = new float2[obstacleCount];
			for (int i = 0; i < obstacleCount; i++)
			{
				Vector2 centre = obstacles[i].centre;
				Vector2 halfSize = obstacles[i].size * 0.5f;
				obstacleData[i] = new float4(centre.x, centre.y, halfSize.x, halfSize.y);
				float radians = obstacles[i].rotation * Mathf.Deg2Rad;
				rotationData[i] = new float2(Mathf.Cos(radians), Mathf.Sin(radians));
			}

			obstacleBuffer = ComputeHelper.CreateStructuredBuffer<float4>(obstacleCount);
			obstacleBuffer.SetData(obstacleData);
			obstacleRotationBuffer = ComputeHelper.CreateStructuredBuffer<float2>(obstacleCount);
			obstacleRotationBuffer.SetData(rotationData);

			ComputeHelper.SetBuffer(compute, obstacleBuffer, "Obstacles", updatePositionKernel);
			ComputeHelper.SetBuffer(compute, obstacleRotationBuffer, "ObstacleRotations", updatePositionKernel);
			compute.SetInt("obstacleCount", obstacleCount);
		}

		void InitPolygonObstacles()
		{
			int polygonCount = polygonObstacles != null ? polygonObstacles.Length : 0;

			// Release any previous buffers
			ComputeHelper.Release(polygonVertexBuffer, polygonVertexCountBuffer);

			if (polygonCount == 0)
			{
				compute.SetInt("polygonObstacleCount", 0);
				return;
			}

			// Allocate buffers
			int totalVertexSlots = polygonCount * MaxPolygonVertices;
			polygonVertexBuffer = ComputeHelper.CreateStructuredBuffer<float2>(totalVertexSlots);
			polygonVertexCountBuffer = ComputeHelper.CreateStructuredBuffer<int>(polygonCount);

			var vertexData = new float2[totalVertexSlots];
			var vertexCounts = new int[polygonCount];

			for (int i = 0; i < polygonCount; i++)
			{
				int count = polygonObstacles[i].vertices != null ? Mathf.Min(polygonObstacles[i].vertices.Length, MaxPolygonVertices) : 0;
				vertexCounts[i] = Mathf.Max(0, count);

				for (int v = 0; v < MaxPolygonVertices; v++)
				{
					int dstIndex = i * MaxPolygonVertices + v;
					if (v < count)
					{
						Vector2 p = polygonObstacles[i].vertices[v];
						vertexData[dstIndex] = new float2(p.x, p.y);
					}
					else
					{
						// Repeat last valid vertex (or zero if none)
						if (count > 0)
						{
							Vector2 p = polygonObstacles[i].vertices[count - 1];
							vertexData[dstIndex] = new float2(p.x, p.y);
						}
						else
						{
							vertexData[dstIndex] = float2.zero;
						}
					}
				}
			}

			polygonVertexBuffer.SetData(vertexData);
			polygonVertexCountBuffer.SetData(vertexCounts);

			ComputeHelper.SetBuffer(compute, polygonVertexBuffer, "PolygonVertices", updatePositionKernel);
			ComputeHelper.SetBuffer(compute, polygonVertexCountBuffer, "PolygonVertexCounts", updatePositionKernel);
			compute.SetInt("polygonObstacleCount", polygonCount);
			compute.SetInt("polygonMaxVertices", MaxPolygonVertices);
		}

#if UNITY_EDITOR
		void OnValidate()
		{
			// Keep obstacle buffer in sync in the editor when values change
			if (Application.isPlaying)
			{
				InitObstacles();
			}
		}
#endif
	}
}