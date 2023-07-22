import numpy as np

def create_icosphere( position=[0,0,0], radius=1, subdivisions=1):
    """ Function that create an icosphere of a given radius and in a given position.
    Input:
        position .- array
        radius .- float
        subdivisions .- int
    Output:
        V, F list of vertices and faces
    """
    position = np.array(position)

    # Define the icosahedron vertices
    a = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [-1, a, 0],
        [1, a, 0],
        [-1, -a, 0],
        [1, -a, 0],
        [0, -1, a],
        [0, 1, a],
        [0, -1, -a],
        [0, 1, -a],
        [a, 0, -1],
        [a, 0, 1],
        [-a, 0, -1],
        [-a, 0, 1]
    ])

    # Normalize the vertices to the sphere of the given radius
    vertices /= np.linalg.norm(vertices, axis=1)[:, np.newaxis]
    

    # Define the icosahedron faces
    faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1]
    ])

    # Subdivide the faces to create a sphere
    for _ in range(subdivisions):
        new_faces = []
        for face in faces:
            # Vertices indices of the face
            a, b, c = face

            # Mid points of the sides
            ab = (vertices[a] + vertices[b]) / 2
            bc = (vertices[b] + vertices[c]) / 2
            ca = (vertices[c] + vertices[a]) / 2

            # Normalize to the unit sphere
            ab = ab/np.linalg.norm(ab)
            bc = bc/np.linalg.norm(bc)
            ca = ca/np.linalg.norm(ca)

            # Add new vertices
            vertices = np.vstack([vertices, ab, bc, ca])

            # Indices of the new vertices
            ab_idx = len(vertices) - 3
            bc_idx = len(vertices) - 2
            ca_idx = len(vertices) - 1

            # Add new faces
            new_faces.append([a, ab_idx, ca_idx])
            new_faces.append([ab_idx, b, bc_idx])
            new_faces.append([ca_idx, bc_idx, c])
            new_faces.append([ab_idx, bc_idx, ca_idx])

        faces = np.array(new_faces)
    
    # Scale vertices to the given radius
    vertices *= radius
    # Translate vertices to position
    vertices += position

    # Return the vertices and faces
    return vertices, faces