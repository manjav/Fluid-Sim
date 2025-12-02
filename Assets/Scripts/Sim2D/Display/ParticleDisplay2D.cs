using Seb.Fluid2D.Simulation;
using Seb.Helpers;
using UnityEngine;

namespace Seb.Fluid2D.Rendering
{
    public class ParticleDisplay2D : MonoBehaviour
    {
        public FluidSim2D sim;
        public Mesh mesh;
        public Shader shader;

        [SerializeField] private float _scale;
        public float scale
        {
            get => _scale;
            set
            {
                if (_scale != value)
                {
                    _scale = value;
                    MarkDirty();
                }
            }
        }

        [SerializeField] private Gradient _colourMap;
        public Gradient colourMap
        {
            get => _colourMap;
            set
            {
                _colourMap = value;
                MarkDirty();
            }
        }

        [SerializeField] private int _gradientResolution;
        public int gradientResolution
        {
            get => _gradientResolution;
            set
            {
                if (_gradientResolution != value)
                {
                    _gradientResolution = value;
                    MarkDirty();
                }
            }
        }

        [SerializeField] private float _velocityDisplayMax;
        public float velocityDisplayMax
        {
            get => _velocityDisplayMax;
            set
            {
                if (_velocityDisplayMax != value)
                {
                    _velocityDisplayMax = value;
                    MarkDirty();
                }
            }
        }

        [Header("Rendering Camera")]
        public Camera targetCamera;
        public bool showInScene;

        Material material;
        ComputeBuffer argsBuffer;
        Bounds bounds;
        Texture2D gradientTexture;

        bool needsUpdate = true;

        void OnEnable()
        {
            // Ensure material exists in the editor when tweaking values (and in play)
            if (material == null && shader != null)
                material = new Material(shader);

            // ensure we update at least once after enable
            needsUpdate = true;
        }

        void Start()
        {
            if (material == null && shader != null)
                material = new Material(shader);

            needsUpdate = true;
        }

        void LateUpdate()
        {
            if (shader == null || sim == null || mesh == null) return;

            UpdateSettings();

            Graphics.DrawMeshInstancedIndirect(
                mesh,
                0,
                material,
                bounds,
                argsBuffer,
                0,
                null,
                UnityEngine.Rendering.ShadowCastingMode.Off,
                false,
                0,
                targetCamera
            );

            if (showInScene)
                Graphics.DrawMeshInstancedIndirect(mesh, 0, material, bounds, argsBuffer);
        }

        void UpdateSettings()
        {
            // safety: sim buffers must exist
            if (sim == null || sim.positionBuffer == null) return;

            material.SetBuffer("Positions2D", sim.positionBuffer);
            material.SetBuffer("Velocities", sim.velocityBuffer);
            material.SetBuffer("DensityData", sim.densityBuffer);

            ComputeHelper.CreateArgsBuffer(ref argsBuffer, mesh, sim.positionBuffer.count);
            bounds = new Bounds(Vector3.zero, Vector3.one * 10000);

            if (!needsUpdate) return;
            needsUpdate = false;

            TextureFromGradient(ref gradientTexture, gradientResolution, colourMap);
            material.SetTexture("ColourMap", gradientTexture);

            material.SetFloat("scale", scale);
            material.SetFloat("velocityMax", velocityDisplayMax);
        }

        public static void TextureFromGradient(ref Texture2D texture, int width, Gradient gradient,
            FilterMode filterMode = FilterMode.Bilinear)
        {
            if (texture == null)
            {
                texture = new Texture2D(width, 1);
            }
            else if (texture.width != width)
            {
                texture.Reinitialize(width, 1);
            }

            if (gradient == null)
            {
                gradient = new Gradient();
                gradient.SetKeys(
                    new GradientColorKey[] { new GradientColorKey(Color.black, 0), new GradientColorKey(Color.black, 1) },
                    new GradientAlphaKey[] { new GradientAlphaKey(1, 0), new GradientAlphaKey(1, 1) }
                );
            }

            texture.wrapMode = TextureWrapMode.Clamp;
            texture.filterMode = filterMode;

            Color[] cols = new Color[width];
            for (int i = 0; i < cols.Length; i++)
            {
                float t = i / (cols.Length - 1f);
                cols[i] = gradient.Evaluate(t);
            }

            texture.SetPixels(cols);
            texture.Apply();
        }

        void MarkDirty()
        {
            needsUpdate = true;
        }

#if UNITY_EDITOR
        // Editor-only: fires when inspector values change — this writes directly
        // to the serialized fields and so is required to catch Inspector edits.
        void OnValidate()
        {
            // Make sure material exists in editor when tweaking
            if (material == null && shader != null)
                material = new Material(shader);

            MarkDirty();
        }
#endif

        void OnDestroy()
        {
            ComputeHelper.Release(argsBuffer);
        }
    }
}
