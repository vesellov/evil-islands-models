import numpy as np


def axisAngleToQuaternion(axis:np.ndarray, angle:float) -> np.ndarray:
    """Creates a quaternion from the given rotation parameters (axis and angle in radians).

    Parameters
    ----------
    axis : [3x1] np.ndarray
            rotation axis
    angle : float
            rotation angle in radians

    Returns
    -------
        [4x1] np.ndarray
        quaternion defining the orientation    
    """
    if isinstance(axis, list) and len(axis)==3:
        axis = np.array(axis) 
    elif isinstance(axis, np.ndarray) and axis.size==3:
        pass
    else:
        raise TypeError("The axis of rotation must be given as [3x1] np.ndarray vector or a python list of 3 elements")
    if angle == 0:
        p = np.array([1,0,0,0])  #  identity quaternion
    else:     
        if np.linalg.norm(axis) == 0:
            raise Exception("A valid rotation 'axis' parameter must be provided to describe a meaningful rotation.")
        else:
            u = axis/np.linalg.norm(axis)    
            p = np.append(np.cos(angle/2), np.sin(angle/2)*u)
    return p


def quaternionToAxisAngle(p:np.ndarray) -> list:
    """Compute rotation parameters (axis and angle in radians) from a quaternion p defining a given orientation.

    Parameters
    ----------
        [4x1] np.ndarray
        quaternion defining a given orientation

    Returns
    -------
    axis : [3x1] np.ndarray, when undefined=[0. 0. 0.]
    angle : float      
    """
    if isinstance(p, list) and len(p)==4:
        e0 = np.array(p[0])
        e = np.array(p[1:])  
    elif isinstance(p, np.ndarray) and p.size==4:
        e0 = p[0]
        e = p[1:]
    else:
        raise TypeError("The quaternion \"p\" must be given as [4x1] np.ndarray quaternion or a python list of 4 elements")    
    if np.linalg.norm(e) == 0:
        axis = np.array([1,0,0]) #To be checked again
        angle = 0
    elif np.linalg.norm(e) != 0:
        axis = e/np.linalg.norm(e) 
        if e0 == 0:
            angle = np.pi
        else:
            angle = 2*np.arctan(np.linalg.norm(e)/e0) 
    else:
        raise ValueError('Unexpected value')
    return axis, angle


def quaternionToRotationMatrix(p:np.ndarray) -> np.ndarray:
    """Computes the rotation matrix given a quaternion p defining an orientation.

    Parameters
    ----------
        [4x1] np.ndarray
        quaternion defining a given orientation.  

    Returns
    -------
    rotation_matrix : [3x3] np.ndarray
    """   
    if isinstance(p, list) and len(p)==4:
        p = np.array(p) 
    elif isinstance(p, np.ndarray) and p.size==4:
        pass
    else:
        raise TypeError("The quaternion must be given as [4x1] np.ndarray vector or a python list of 4 elements")
    e0 = p[0]
    e1 = p[1]
    e2 = p[2]
    e3 = p[3]

    # rotation_matrix = np.array([[e0*e0 + e1*e1 - e2*e2 - e3*e3, 2*e1*e2 - 2*e0*e3, 2*e0*e2 + 2*e1*e3],
    #                             [2*e0*e3 + 2*e1*e2, e0*e0 - e1*e1 + e2*e2 - e3*e3, 2*e2*e3 - 2*e0*e1],
    #                             [2*e1*e3 - 2*e0*e2, 2*e0*e1 + 2*e2*e3, e0*e0 - e1*e1 - e2*e2 + e3*e3]])        

    rotation_matrix = [
        [e0*e0 + e1*e1 - e2*e2 - e3*e3, 2*e1*e2 - 2*e0*e3, 2*e0*e2 + 2*e1*e3, 0.0],
        [2*e0*e3 + 2*e1*e2, e0*e0 - e1*e1 + e2*e2 - e3*e3, 2*e2*e3 - 2*e0*e1, 0.0],
        [2*e1*e3 - 2*e0*e2, 2*e0*e1 + 2*e2*e3, e0*e0 - e1*e1 - e2*e2 + e3*e3, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return rotation_matrix


def rotationMatrixToQuaternion(R:np.ndarray) -> np.ndarray:
    """Creates a quaternion from a rotation matrix defining a given orientation.

    Parameters
    ----------
    R : [3x3] np.ndarray
        Rotation matrix
    
    Returns
    -------
        [4x1] np.ndarray
        quaternion defining the orientation    
    """    
    u_q0 = np.sqrt((1 + R[0,0] + R[1,1] + R[2,2])/4) # the prefix u_ means unsigned
    u_q1 = np.sqrt((1 + R[0,0] - R[1,1] - R[2,2])/4)
    u_q2 = np.sqrt((1 - R[0,0] + R[1,1] - R[2,2])/4)
    u_q3 = np.sqrt((1 - R[0,0] - R[1,1] + R[2,2])/4)

    q = np.array([u_q0, u_q1, u_q2, u_q3])

    if u_q0 == max(q):
        q0 = u_q0
        q1 = (R[2,1] - R[1,2])/(4*q0)
        q2 = (R[0,2] - R[2,0])/(4*q0)
        q3 = (R[1,0] - R[0,1])/(4*q0)

    if u_q1 == max(q):
        q1 = u_q1
        q0 = (R[2,1] - R[1,2])/(4*q1)
        q2 = (R[0,1] + R[1,0])/(4*q1)
        q3 = (R[0,2] + R[2,0])/(4*q1)

    if u_q2 == max(q):
        q2 = u_q2
        q0 = (R[0,2] - R[2,0])/(4*q2)
        q1 = (R[0,1] + R[1,0])/(4*q2)
        q3 = (R[1,2] + R[2,1])/(4*q2)    

    if u_q3 == max(q):
        q3 = u_q3
        q0 = (R[1,0] - R[0,1])/(4*q3)  
        q1 = (R[0,2] + R[2,0])/(4*q3)
        q2 = (R[1,2] + R[2,1])/(4*q3)  

    return np.array([q0, q1, q2, q3])


def eulerAnglesToQuaternion(eulerAngles:np.ndarray|list)->np.ndarray:
    """
    Convert an Euler angle to a quaternion.

    We have used the following definition of Euler angles.

    - Tait-Bryan variant of Euler Angles
    - Yaw-pitch-roll rotation order (ZYX convention), rotating around the z, y and x axes respectively
    - Intrinsic rotation (the axes move with each rotation)
    - Active (otherwise known as alibi) rotation (the point is rotated, not the coordinate system)
    - Right-handed coordinate system with right-handed rotations

    Parameters
    ----------
    eulerAngles : 
        [3x1] np.ndarray  
        [roll, pitch, yaw] angles in radians 
    
    Returns
    -------
        [4x1] np.ndarray
        quaternion defining a given orientation
    """
    if isinstance(eulerAngles, list) and len(eulerAngles)==3:
        eulerAngles = np.array(eulerAngles) 
    elif isinstance(eulerAngles, np.ndarray) and eulerAngles.size==3:
        pass
    else:
        raise TypeError("The eulerAngles must be given as [3x1] np.ndarray vector or a python list of 3 elements")

    roll = eulerAngles[0]
    pitch = eulerAngles[1]
    yaw = eulerAngles[2]

    q0 = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    q1 = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    q2 = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    q3 = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)

    return np.r_[q0, q1, q2, q3]


def quaternionToEulerAngles(q:np.ndarray|list)->np.ndarray:
    """
    Convert a quaternion into euler angles [roll, pitch, yaw]
    - roll is rotation around x in radians (CCW)
    - pitch is rotation around y in radians (CCW)
    - yaw is rotation around z in radians (CCW)

    Parameters
    ----------
    q : [4x1] np.ndarray
        quaternion defining a given orientation
  
    Returns
    -------
        [3x1] np.ndarray  
        [roll, pitch, yaw] angles in radians 
    """
    if isinstance(q, list) and len(q)==4:
        q = np.array(q) 
    elif isinstance(q, np.ndarray) and q.size==4:
        pass
    else:
        raise TypeError("The quaternion must be given as [4x1] np.ndarray vector or a python list of 4 elements")

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    t2 = 2.0*(q0*q2 - q1*q3)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2

    if t2 == 1:
        pitch = np.arcsin(t2)
        roll = 0
        yaw = -np.arctan2(q0, q1)
    elif t2 == -1:
        pitch = np.arcsin(t2)
        roll = 0
        yaw = +np.arctan2(q0, q1)
    else:
        pitch = np.arcsin(t2)
        roll = np.arctan2(2.0*(q0*q1 + q2*q3), q0*q0 - q1*q1 - q2*q2 + q3*q3)
        yaw = np.arctan2(2.0*(q0*q3 + q1*q2), q0*q0 + q1*q1 - q2*q2 - q3*q3)

    return np.r_[roll, pitch, yaw]


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    # return np.array([
    #     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
    #     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
    #     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
    #     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
    # ], dtype=np.float64)
    return [
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
    ]


def trilinear(val, coefs=[0, 0, 0]):
    # Linear interpolation by str
    t1 = val[0] + (val[1] - val[0]) * coefs[1]
    t2 = val[2] + (val[3] - val[2]) * coefs[1]
    # Bilinear interpolation by dex
    v1 = t1 + (t2 - t1) * coefs[0]
    # Linear interpolation by str
    t1 = val[4] + (val[5] - val[4]) * coefs[1]
    t2 = val[6] + (val[7] - val[6]) * coefs[1]
    # Bilinear interpolation by dex
    v2 = t1 + (t2 - t1) * coefs[0]
    # Trilinear interpolation by height
    return v1 + (v2 - v1) * coefs[2]
