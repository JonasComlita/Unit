import { extractVertexIdFromMeshName } from '../gameboardUtils';

describe('extractVertexIdFromMeshName', () => {
  it('returns null for empty string', () => {
    expect(extractVertexIdFromMeshName('')).toBeNull();
  });

  it('extracts vertex id from plain vertex name', () => {
    expect(extractVertexIdFromMeshName('vertex-12')).toBe('vertex-12');
  });

  it('extracts vertex id from indicator mesh name', () => {
    expect(extractVertexIdFromMeshName('indicator-vertex-8')).toBe('vertex-8');
  });

  it('extracts vertex id from material name', () => {
    expect(extractVertexIdFromMeshName('mat-tap-vertex-110')).toBe('vertex-110');
  });

  it('returns null when no vertex id present', () => {
    expect(extractVertexIdFromMeshName('piece_')).toBeNull();
    expect(extractVertexIdFromMeshName('random-mesh-name')).toBeNull();
  });
});
