// Native OpenUSD in Unity, through Unity's own USD Toolkit (com.unity.usd-toolkit, Unity 6.4+).
//
// PINNeAPPle writes a .usda that carries everything this importer needs: parts as UsdGeom.Mesh, every field as a
// per-vertex primvars:<name> with time samples, UsdPhysics colliders/rigid bodies/materials and the provenance in the
// layer metadata. Importing it here gives Unity the geometry with its materials and textures, the animation as a
// stage (UsdAnimationPlayer), and -- with ImportPhysics on -- rigidbodies, colliders and ArticulationBody chains
// driven by the stage's own limits and drives. Exporting the other way writes a stage Isaac Sim or Omniverse opens.
//
// The calls are the toolkit's documented API (0.8.x). The file compiles in two modes: with the package installed
// (the asmdef's versionDefines sets PINNEAPPLE_USD_TOOLKIT) it imports and exports; without it, the component says
// what to install instead of failing at compile time.
#if PINNEAPPLE_USD_TOOLKIT
using Unity.USDToolkit;
#endif
using System.IO;
using UnityEngine;

namespace PinNeAPPle.Unity
{
    [AddComponentMenu("PINNeAPPle/USD Stage")]
    public class PinneappleUsdStage : MonoBehaviour
    {
        [Header("Import")]
        [Tooltip("A .usda/.usd/.usdc/.usdz stage: PINNeAPPle's Assets/TwinData/scene.usda, or one from Isaac Sim.")]
        public string usdPath = "Assets/TwinData/scene.usda";
        public bool importOnStart = false;
        public bool importMaterials = true;
        public bool importTextures = true;
        [Tooltip("UsdPhysics -> rigidbodies, colliders and ArticulationBody chains (toolkit 0.8+).")]
        public bool importPhysics = true;
        [Tooltip("The toolkit applies the stage's up axis and metersPerUnit to the import root.")]
        public bool applyStageUpAxisAndUnits = true;
        [Tooltip("Leave empty for a stage that composes everything; list variants to pick a configuration.")]
        public string[] variantSelections = new string[0];

        [Header("Export")]
        [Tooltip("Root to write. Empty: this GameObject.")]
        public Transform exportRoot;
        public string exportPath = "";
        public bool exportTextures = true;
        [Tooltip("Preserve keeps references, payloads and variants instead of flattening.")]
        public bool preserveComposition = true;

        [Header("Result")]
        public int importedMeshes;
        public long importedTriangles;
        public string lastMessage = "";

        public UsdImportPreviewInfo Preview()
        {
#if PINNEAPPLE_USD_TOOLKIT
            string path = Resolve(usdPath);
            UsdImportPreviewInfo preview = UsdImporter.GetPreviewInfo(path);
            Debug.Log($"PINNeAPPle USD: {Path.GetFileName(path)} -- {preview.MeshCount} mesh(es), " +
                      $"{preview.TriangleCount} triangle(s), up axis {preview.UpAxis}", this);
            importedMeshes = preview.MeshCount;
            importedTriangles = preview.TriangleCount;
            return preview;
#else
            Missing();
            return default;
#endif
        }

        public async void Import()
        {
#if PINNEAPPLE_USD_TOOLKIT
            string path = Resolve(usdPath);
            if (!File.Exists(path))
            {
                Debug.LogError("PINNeAPPle USD: no stage at " + path, this);
                return;
            }
            UsdImportPreviewInfo preview = UsdImporter.GetPreviewInfo(path);
            importedMeshes = preview.MeshCount;
            importedTriangles = preview.TriangleCount;
            Debug.Log($"PINNeAPPle USD: importing {Path.GetFileName(path)} -- {preview.MeshCount} mesh(es), " +
                      $"{preview.TriangleCount} triangle(s), up axis {preview.UpAxis}, physics {(importPhysics ? "on" : "off")}",
                      this);
            UsdImportResult result = await UsdImporter.ImportAsync(path, new UsdImportOptions
            {
                Parent = transform,
                ImportMaterials = importMaterials,
                ImportTextures = importTextures,
                ImportPhysics = importPhysics,
                ApplyStageUpAxisAndUnits = applyStageUpAxisAndUnits,
                VariantSelections = variantSelections,
                LoadPayloads = true,
            });
            await result.TexturesReady;                        // textures keep streaming in after the meshes
            lastMessage = result.ToString();
            if (result.PhysicsNotes != null)
                foreach (string note in result.PhysicsNotes) Debug.Log("PINNeAPPle USD physics: " + note, this);
            Debug.Log("PINNeAPPle USD: " + lastMessage, this);
#else
            Missing();
            await System.Threading.Tasks.Task.CompletedTask;
#endif
        }

        public void Export()
        {
#if PINNEAPPLE_USD_TOOLKIT
            Transform root = exportRoot != null ? exportRoot : transform;
            string path = string.IsNullOrEmpty(exportPath)
                ? Path.Combine(Application.persistentDataPath, "pinneapple_scene.usda")
                : Resolve(exportPath);
            UsdExportResult result = UsdExporter.ExportGameObjectWithResult(root, path, new UsdExportOptions
            {
                TransformPolicy = UsdTransformPolicy.PreserveHierarchy,
                ExportTextures = exportTextures,
                Composition = preserveComposition ? UsdCompositionMode.Preserve : UsdCompositionMode.Flatten,
            });
            lastMessage = result.ToString();
            Debug.Log("PINNeAPPle USD: wrote " + path + " -- " + lastMessage +
                      ". `python -m pinneapple_unity ingest` reads it back and runs the jobs.", this);
#else
            Missing();
#endif
        }

        void Start()
        {
            if (importOnStart) Import();
        }

        /// <summary>A path as written is project-relative (Assets/...), not process-relative.</summary>
        string Resolve(string path)
        {
            if (string.IsNullOrEmpty(path) || Path.IsPathRooted(path)) return path;
            string project = Directory.GetParent(Application.dataPath).FullName;
            return Path.Combine(project, path);
        }

        void Missing()
        {
            const string how = "install Unity's USD Toolkit (Unity 6.4+): Package Manager -> + -> Add package from git URL -> " +
                               "https://github.com/Unity-Technologies/com.unity.usd-toolkit.git#v0.8.1-exp.1 " +
                               "(python -m pinneapple_unity.usd manifest <project> writes the same line into " +
                               "Packages/manifest.json)";
            Debug.LogError("PINNeAPPle USD: this project has no USD Toolkit -- " + how, this);
        }
    }
}
