// The shared three.js field viewer of the PINNeAPPle apps (/shared/fieldview3d.js) on a scene written by the
// viz.field3d node (pp.viz.field_view: glTF + pinneapple-field-view/v1 meta).
import { useEffect, useRef, useState } from "react";
import { artifactUrl } from "../api";

export function Viewer3D({ scene }: { scene: { glb: string; meta: any } }) {
  const host = useRef<HTMLDivElement>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    let cancelled = false;
    const src = "/shared/fieldview3d.js";
    import(/* @vite-ignore */ src)
      .then((m: any) => {
        if (cancelled || !host.current) return;
        new m.FieldView3D(host.current, { meta: scene.meta, glb: artifactUrl(scene.glb), height: 360 });
      })
      .catch((e) => setErr(`3-D viewer unavailable: ${e}`));
    return () => {
      cancelled = true;
      if (host.current) host.current.innerHTML = "";
    };
  }, [scene.glb]);
  return (
    <div>
      {err && <div className="text-xs text-red-600">{err}</div>}
      <div ref={host} data-testid="viewer3d" />
    </div>
  );
}
