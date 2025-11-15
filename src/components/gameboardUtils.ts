// src/components/gameboardUtils.ts
export function extractVertexIdFromMeshName(name: string): string | null {
  if (!name) return null;
  const match = name.match(/vertex-\d+/);
  return match ? match[0] : null;
}

export default extractVertexIdFromMeshName;
