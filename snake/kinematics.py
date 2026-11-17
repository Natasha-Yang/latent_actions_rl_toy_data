"""Planar kinematics for a free-floating snake made of N+1 rigid links and N joints.

Configuration: tail pose (x, y, phi) plus joint angles q (N,). Link 0 starts at
the tail with heading phi; joint j bends link j+1 by q[j] relative to link j:
    heading_k = phi + sum_{j<k} q_j      (k = 0..N)
"""
import numpy as np


def fk(q, link_len, base=(0.0, 0.0), base_angle=0.0):
    """Positions of the tail, the N joints and the head: an (N+2, 2) array
    ordered tail -> head."""
    q = np.asarray(q, dtype=np.float64)
    theta = base_angle + np.concatenate([[0.0], np.cumsum(q)])
    seg = link_len * np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    pts = np.empty((q.shape[0] + 2, 2))
    pts[0] = base
    pts[1:] = np.asarray(base, dtype=np.float64) + np.cumsum(seg, axis=0)
    return pts


def chain_from_points(pts):
    """Inverse of fk for points spaced ~link_len apart: returns (base, phi, q).

    Link headings are taken from consecutive point differences, so any points
    lying on a smooth curve give a chain whose joints lie (almost) on it.
    """
    d = np.diff(pts, axis=0)
    heading = np.arctan2(d[:, 1], d[:, 0])
    return pts[0].copy(), heading[0], wrap(np.diff(heading))


def wrap(a):
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
