import numpy as np
import os

def fill_mesh_holes_simple(vertices, faces):
    """
    Attempts to fill holes in a mesh by creating new faces.

    This is a very basic and limited approach that only works for simple,
    planar holes. It's not as robust as trimesh's fill_holes.

    Args:
        vertices (numpy.ndarray): Array of vertex coordinates (N, 3).
        faces (numpy.ndarray): Array of face indices (M, 3).

    Returns:
        tuple: (updated vertices, updated faces)
    """

    def find_boundary_edges(faces):
        """Finds edges that are only used by one face (boundary edges)."""
        edge_counts = {}
        for face in faces:
            for i in range(3):
                edge = tuple(sorted((face[i], face[(i + 1) % 3])))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

        boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
        return boundary_edges

    def find_boundary_loop(boundary_edges):
        """Attempts to reconstruct a boundary loop from boundary edges."""
        if not boundary_edges:
            return []

        loop = [boundary_edges[0][0]]
        remaining_edges = boundary_edges[:]

        while True:
            found_next = False
            for edge in remaining_edges:
                if edge[0] == loop[-1]:
                    loop.append(edge[1])
                    remaining_edges.remove(edge)
                    found_next = True
                    break
                elif edge[1] == loop[-1]:
                    loop.append(edge[0])
                    remaining_edges.remove(edge)
                    found_next = True
                    break

            if not found_next:
                break

            if loop[0] == loop[-1] and len(loop)>3: #close loop
                return loop[:-1] #remove duplicate last element.

        return []  # Failed to find a closed loop

    boundary_edges = find_boundary_edges(faces)
    boundary_loop = find_boundary_loop(boundary_edges)

    if not boundary_loop:
        return vertices, faces  # No boundary loop found, nothing to fill

    # Create a new face to fill the hole (simplest case: triangle fan)
    if len(boundary_loop) >= 3:
        new_faces = []
        for i in range(1, len(boundary_loop) - 1):
            new_faces.append([boundary_loop[0], boundary_loop[i], boundary_loop[i + 1]])

        updated_faces = np.concatenate((faces, np.array(new_faces)), axis=0)
        return vertices, updated_faces
    else:
        return vertices, faces # Boundary loop too small.

def load_obj(filepath):
    """Loads vertices and faces from an OBJ file."""
    vertices = []
    faces = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:])))
            elif line.startswith('f '):
                face_indices = [int(x.split('/')[0]) - 1 for x in line.split()[1:]] #handle obj face indexing
                faces.append(face_indices)
    return np.array(vertices), np.array(faces)

def save_obj(filepath, vertices, faces):
    """Saves vertices and faces to an OBJ file."""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            f.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n') #obj face indexing

#Example Usage
if __name__ == "__main__":
    input_obj = "input_mesh_with_holes.obj"
    output_obj = "filled_mesh_simple.obj"

    try:
        if not os.path.exists(input_obj): #create a sample object if it doesn't exist.
            vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1], [2,0,0]])
            faces = np.array([[0, 1, 2], [0, 2, 3], [4,0,1], [4,1,2]]) #create a hole.
            save_obj(input_obj, vertices, faces)
            print(f"Created sample object: {input_obj}")

        vertices, faces = load_obj(input_obj)
        filled_vertices, filled_faces = fill_mesh_holes_simple(vertices, faces)
        save_obj(output_obj, filled_vertices, filled_faces)
        print(f"Filled mesh saved to: {output_obj}")
    except FileNotFoundError:
        print(f"File not found: {input_obj}")