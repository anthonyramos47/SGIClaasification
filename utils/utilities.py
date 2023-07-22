import numpy as np

# Sparse functions
from scipy.sparse.linalg import eigs
from scipy.sparse import coo_matrix, diags


# Visualization Shading options
shading = {"flat":True, # Flat or smooth shading of triangles
           "wireframe":True, "wire_width": 0.05, "wire_color": "black", # Wireframe rendering
           "width": 600, "height": 600, # Size of the viewer canvas
           "antialias": True, # Antialising, might not work on all GPUs
           "scale": 2.0, # Scaling of the model
           "side": "DoubleSide", # FrontSide, BackSide or DoubleSide rendering of the triangles
           "background": "#ffffff", # Background color of the canvas
           "line_width": 1.5, "line_color": "black", # Line properties of overlay lines
           "bbox": False, # Enable plotting of bounding box
           "point_color": "red", "point_size": 0.01 # Point properties of overlay points
          }

def face_areas(v,f):
    """ Function that compute the face areas
    """
    # Get number of faces
    m = np.shape(f)[0]

    # Get number of vertices
    n = np.shape(v)[0]

    # Get vertex indices
    i1, i2, i3 = f[:,0], f[:,1], f[:,2]

    # Get vertices
    v1, v2, v3 = v[i1,:], v[i2,:], v[i3,:]

    # Get edges
    e1 = (v2 - v1)
    e2 = (v3 - v1)

    # Compute cross product 
    A = np.cross(e1, e2)

    A = 0.5*np.linalg.norm(A, axis=1)

    return A

def barycentric_area(v,f):
    """ Function that compute the barycentric area
    """
    # Get number of vertices
    n = np.shape(v)[0]

    # Get the list of areas
    face_A = face_areas(v,f)

    # Vertex Bar Areas
    bar_A = np.empty((n,1))

    for i in range(n):
        idx_f = np.where(f == i)[0]

        bar_A[i] = np.sum(face_A[idx_f])/3

    return bar_A

def cotangent_matrix(v,f):
    """ Function that compute the cotangent matrix given
    the vertices and the faces list of a mesh. 
    Output: L.- sparse laplacian matrix
    """
    # Get number of faces
    m = np.shape(f)[0]

    # Get number of vertices
    n = np.shape(v)[0]

    values = []
    # I and J indices i<->j of an edge
    I = np.empty( (0,), dtype=int)
    J = np.empty( (0,), dtype=int)

    # Get vertex indices
    i1, i2, i3 = f[:,0], f[:,1], f[:,2]

    # Get vertices
    v1, v2, v3 = v[i1,:], v[i2,:], v[i3,:]

    # Get normlaized edges
    e1 = (v2 - v1)
    e2 = (v3 - v1)
    e3 = (v3 - v2)

    # Compute the lenght of the edges
    le1 = np.linalg.norm(e1, axis=1)
    le2 = np.linalg.norm(e2, axis=1) 
    le3 = np.linalg.norm(e3, axis=1)

    # Normalize the edges
    e1 = e1 / le1[:, None]
    e2 = e2 / le2[:, None]
    e3 = e3 / le3[:, None]
    #       e1   
    #  v1 ------ v2 
    #    \      /
    #  e2 \    / e3   
    #       v3

    #denom12 = np.sqrt(np.sum(1 - (e2*e3)**2, axis=1))
    #denom23 = np.sqrt(np.sum(1 - (e2*e1)**2, axis=1))
    #denom13 = np.sqrt(np.sum(1 - (e1*e3)**2, axis=1))
    # Cosine values for the triangle
    arcos12 = np.arccos(np.sum(e2*e3, axis=1) ) 
    arcos23 = np.arccos(np.sum(e1*e2, axis=1) ) 
    arcos13 = np.arccos(np.sum(-e1*e3, axis=1) ) 

    cota12 =  0.5*(-np.tan(arcos12 + np.pi/2))
    cota23 =  0.5*(-np.tan(arcos23 + np.pi/2))
    cota13 =  0.5*(-np.tan(arcos13 + np.pi/2))


    # Stack indices of the triangles
    I = np.hstack((I, i1, i2, i3))
    J = np.hstack((J, i2, i3, i1))

    # Stack values of the cosines angles
    values = np.hstack((values, cota12, cota23, cota13))

    # Create sparse matrix half laplacian
    hl = coo_matrix((values, (I, J)))

   
    # Off diagonal laplacian
    L = hl + hl.transpose()

    # Sum rows
    diagonal =  np.array(- L.sum(axis=1))

    diagonal_L = diags(diagonal.T[0], offsets=0 )    

    L = L + diagonal_L
    return L

def inv_mass_matrix(v,f):
    """Function that return a mass matrix
    """
    
    bar_areas = 1/barycentric_area(v,f)

    n = np.shape(v)[0]

    print(bar_areas.T)

    M = diags(bar_areas.T[0], offsets=0, shape=(n,n) )
    
    return M
         

def normalize_row(lv):
  """ Function that given a list of vertices return the vertices normalized
  """

  norms = np.linalg.norm(lv, axis=1)

  unit_v = lv / norms[:, None]

  return unit_v



def adjacent_faces(v,f):
  """ Function that return a matrix witht the indices of adjacent faces to a vertex
  """
  # Number of vertices
  n = np.shape(v)[0]

  # Initialize dic 
  adj = {}

  for i in range(n):
    adj[i]= np.where(f == i)[0]

  return adj

def circumcircle_faces(v,f):
  """ Function that return a matrix with the center of the circumcircle of each face
  """
  # Get number of faces
  m = np.shape(f)[0]

  # Get number of vertices
  n = np.shape(v)[0]

  # Get vertex indices
  i1, i2, i3 = f[:,0], f[:,1], f[:,2]

  # Get vertices
  v1, v2, v3 = v[i1,:], v[i2,:], v[i3,:]

  # Get edges
  e1 = v2 - v1
  e2 = v3 - v1
  e3 = v3 - v2

  # Local coordinate frame
  u1 = normalize_row(v2 - v1)
  u2 = normalize_row( np.cross(v3 - v1, u1))
  u3 = np.cross(u2, u1)

  # aux variables
  bx = np.sum(e1*u1, axis=1)
  cx = np.sum(e3*u1, axis=1)
  cy = np.sum(e3*u3, axis=1)

  print(cy[379])

  denom = (2*cy)
  h = ((cx - bx/2)**2 + cy**2 - (bx/2)**2)/denom 

  dispu1 = bx/2

  centers = v1 + dispu1[:, None]*u1 + h[:, None]*u3
  return centers

def compute_ratio(v,f):
  """ Function that return the aspect ratio of each triangle in the mesh.
  """
  # Get vertices indices
  i1, i2, i3 = f[:, 0], f[:, 1], f[:, 2]

  # Get vertices
  v1, v2, v3 = v[i1], v[i2], v[i3]

  # Define the edges lenghts
  e1 = np.linalg.norm(v2 - v1, axis=1)
  e2 = np.linalg.norm(v3 - v1, axis=1)
  e3 = np.linalg.norm(v3 - v2, axis=1)

  lenghts = np.vstack((e1,e2,e3)).transpose()

  # Max lenght
  max_l = np.max(lenghts, axis=1)

  # Circumcircle
  #diameter = 2*(e1*e2*e3)/np.sqrt((e1+e2+e3)*(e2 + e3 - e1)*(e3 + e1 - e2)*(e1 + e2 - e3))

  # Semiperimeter
  s = 0.5*(e1+e2+e3)

  # Radius Inscribed circle
  r = np.sqrt((s-e1)*(s-e2)*(s-e3))

  # sphericity
  sphericity = 2*r

  ratio = max_l/sphericity

  return ratio

def voronoi_area(v,f):
    """ Function that compute the Voronoi Area of each vertex
    Output: V.- Vector of areas
    """

    # Get number of faces
    m = np.shape(f)[0]

    # Get number of vertices
    n = np.shape(v)[0]

    # Get adjacent faces
    adj_v_f = adjacent_faces(v,f)

    return 