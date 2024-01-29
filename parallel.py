import pandas as pd
import os 
import glob 
import math 
import re
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from pprint import pprint
import argparse


# In[171]:


# methods to plot objects in arenas

def plot_arena(ax):
    # arena: arena.world 
    ax.add_patch(patches.Rectangle((-5,-5), 1, 10, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((4,-5), 1, 10, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-5,-5), 10, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-5,4), 10, 1, edgecolor='lightgrey', facecolor='none'))

    
def plot_arena_large(ax):
    # arena: arena_large.world
    ax.add_patch(patches.Rectangle((-10,-10), 1, 20, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((9,-10), 1, 20, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-10,-10), 20, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-10,9), 20, 1, edgecolor='lightgrey', facecolor='none'))

    
def plot_two_rooms(ax): 
    # arena: two_rooms.world
    ax.add_patch(patches.Rectangle((-10,-8), 1, 16, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((9,-8), 1, 16, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-2,2), 1, 6, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-2,-8), 1, 6, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-10,7), 20, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-10,-8), 20, 1, edgecolor='lightgrey', facecolor='none'))

def plot_arena_corners(ax):
    # arena: arena_corners.world 
    ax.add_patch(patches.Rectangle((-5,-6), 1, 12, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((4,-6), 1, 12, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-5,-6), 10, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-5,5), 10, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-4,-3), 1, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((0,4), 1, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-4, 1.5), 1, 1.5, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((3, 0), 1, 2, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((2, -5), 2, 2, edgecolor='lightgrey', facecolor='none'))

def plot_arena_walls(ax):
    # arena: arena_walls.world
    ax.add_patch(patches.Rectangle((-7,-7), 1, 14, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((6,-7), 1, 14, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-7,-7), 14, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-7,6), 14, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-0.125,2), 0.25, 4, edgecolor='lightgrey', facecolor='none')) 
    ax.add_patch(patches.Rectangle((2,-1), 4, 0.25, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-2,-4), 0.25, 3, edgecolor='lightgrey', facecolor='none'))

def plot_turtle(ax):
    # arena: turtle.world 
    
    # ROS LOGO 
    ax.add_patch(patches.Circle((-1.1,-1.1), 0.15, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-1.1,0), 0.15, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-1.1,1.1), 0.15, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((0,-1.1), 0.15, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((0,0), 0.15, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((0,1.1), 0.15, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((1.1,-1.1), 0.15, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((1.1,0), 0.15, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((1.1,1.1), 0.15, edgecolor='lightgrey', facecolor='none'))
    
    # WALLS 
    
    
    # Turtle head etc. 
    

def plot_arena_corners_pillars(ax):
    # arena: arena_corners_pillars.world 
    ax.add_patch(patches.Rectangle((-5,-7), 10, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-5,-7), 1, 14, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-5,6), 10, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((4,-7), 1, 14, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-4,1), 1, 2, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((1,-6), 1, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((2,-6), 1, 2, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((3,-6), 1, 3, edgecolor='lightgrey', facecolor='none'))

    ax.add_patch(patches.Circle((-2,-3), 0.75, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((2,4), 0.75, edgecolor='lightgrey', facecolor='none'))
    

def plot_arena_walls_poles(ax):
    # arena: arena_walls_poles.world 
    ax.add_patch(patches.Rectangle((-7,-7), 1, 14, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((6,-7), 1, 14, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-7,-7), 14, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-7,6), 14, 1, edgecolor='lightgrey', facecolor='none'))    
    ax.add_patch(patches.Rectangle((-6,-1.125), 3, 0.25, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((0.875,2), 0.25, 4, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((5.875,-6), 0.25, 6, angle=45, edgecolor='lightgrey', facecolor='none'))
    
    ax.add_patch(patches.Circle((4,1), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((5,2), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-1,-4), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-2,-4), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-2,4), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-3,3), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-4,2), 0.05, edgecolor='lightgrey', facecolor='none'))

    
def plot_arena_pillars_poles(ax):
    # arena: arena_pillars_poles.world  
    ax.add_patch(patches.Rectangle((-7,3), 14, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((6,-3), 1, 6, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-7,-4), 14, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-7,-3), 1, 6, edgecolor='lightgrey', facecolor='none'))
    
    ax.add_patch(patches.Circle((-4,1), 0.4, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-1,-2), 1.2, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((1,2), 0.3, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((4,-1), 0.75, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((5.5,2.5), 0.5, edgecolor='lightgrey', facecolor='none'))


def plot_arena_pillars(ax):
    # arena: arena_pillars.world 
    ax.add_patch(patches.Rectangle((-10,-10), 1, 20, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((9,-10), 1, 20, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-10,-10), 20, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-10,9), 20, 1, edgecolor='lightgrey', facecolor='none'))
    
    ax.add_patch(patches.Circle((-4.5,4.5), 1.0, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-4.5,-4.5), 1.0, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((4.5,4.5), 1.0, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((4.5,-4.5), 1.0, edgecolor='lightgrey', facecolor='none'))
    
    
def plot_arena_poles(ax):
    # arena: arena_poles.world 
    ax.add_patch(patches.Rectangle((-7,-7), 1, 14, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((6,-7), 1, 14, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-7,-7), 14, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-7,6), 14, 1, edgecolor='lightgrey', facecolor='none')) 
    
    ax.add_patch(patches.Circle((-2,-1), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-2,-2), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-2,-3), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((1,4), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((3,0), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((2,-4), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-2,-1), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((5.9,5.9), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-4,4), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-3,4), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-3,3.5), 0.05, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-4,3.5), 0.05, edgecolor='lightgrey', facecolor='none'))
    

def plot_arena_boxes_pillars(ax):
    # arena: arena_poles.world 
    ax.add_patch(patches.Rectangle((-7,4.5), 14, 1, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-7,-5.5), 14, 1, edgecolor='lightgrey', facecolor='none')) 
    ax.add_patch(patches.Rectangle((-7,-5.5), 1, 11, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((6,-5.5), 1, 11, edgecolor='lightgrey', facecolor='none'))
    
    ax.add_patch(patches.Circle((-0.5,2), 0.7, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Circle((-0.5,-2), 0.7, edgecolor='lightgrey', facecolor='none'))
    
    ax.add_patch(patches.Rectangle((-4,2), 1, 0.5, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-4,0.5), 1, 0.5, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-4,-1), 1, 0.5, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((-4,-2.5), 1, 0.5, edgecolor='lightgrey', facecolor='none'))

    ax.add_patch(patches.Rectangle((2,2), 1, 0.5, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((2,0.5), 1, 0.5, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((2,-1), 1, 0.5, edgecolor='lightgrey', facecolor='none'))
    ax.add_patch(patches.Rectangle((2,-2.5), 1, 0.5, edgecolor='lightgrey', facecolor='none'))

    ax.add_patch(patches.Rectangle((4.25,-1), 0.75, 2, edgecolor='lightgrey', facecolor='none'))


# In[175]:


# method to create visualization 

def create_plot(objects, robots, lidar, arena, r, robot, ax, rot_check, prct):
    # create plot with robots, obstacles and lidar data 
    
    lidar_obj, lidar_kin = lidar
    
    ax.set_title(str(robot)+"  rot_check: "+str(rot_check) + " " +str(prct))
        
    if arena == 'arena':
        plot_arena(ax)
    elif arena == 'arena-large':
        plot_arena_large(ax)
    elif arena == "two-rooms":
        plot_two_rooms(ax) 
    elif arena == "arena-corners":
        plot_arena_corners(ax)
    elif arena == "arena-corners-pillars":
        plot_arena_corners_pillars(ax)
    elif arena == "arena-walls":
        plot_arena_walls(ax)
    elif arena == "arena-walls-poles":
        plot_arena_walls_poles(ax)
    elif arena == "arena-pillars-poles":
        plot_arena_pillars_poles(ax)
    elif arena == "arena-pillars":
        plot_arena_pillars(ax)
    elif arena == "arena-poles":
        plot_arena_poles(ax)
    elif arena == "arena-blocks-pillars":
        plot_arena_blocks_pillars(ax)
    elif arena == "turtle":
        plot_turtle(ax)
    elif arena == "arena-boxes-pillars":
        plot_arena_boxes_pillars(ax)        

    ax.scatter(*objects.T, facecolors='blue', marker='+') # objects except ground plane 
    ax.scatter(*robots.T, facecolors='none', edgecolors='orange') # robots 
    ax.scatter(*r.T, facecolors='green', edgecolors='green', marker='*') # robots 
    ax.scatter(*lidar_obj.T, s=0.5, color='blue') # lidar objects  'red' if rot_check else 
    ax.scatter(*lidar_kin.T, s=0.5, color='orange') # lidar kins  'red' if rot_check else 


# In[176]:


# data frame processing 

def get_positions(df, idx):        
    # get object and robot positions from data frame 
    
    objects = [] 
    robots = [] 
    r = []
    angle = 0 
    
    # get object names  
    names = [el[1:-1] for el in df['poses'][idx].split('[')[1].split(']')[0].split(', ')] 

    string = df['poses'][idx]
    
    for i in range(0, len(names)):
        # get position 
        start = string.find('Point')
        stop = string[start:].find(')')
        pose = string[start+6:start+stop].split(', ')
        
        # get yaw angle (rotation around x-axis )
        string = string[start+6:]
        start = string.find('Quaternion')
        stop = string[start:].find(')')
        orientation = string[start+11:start+stop].split(', ')
        
        if names[i] == df['robot'][idx]: 
            r = [float(pose[0][2:]), float(pose[1][2:])]
            angle = calculate_yaw(orientation)            
        elif names[i][:5] == "robot" :
            robots.append([float(pose[0][2:]), float(pose[1][2:])])             
        elif names[i] != "ground_plane": 
            objects.append([float(pose[0][2:]), float(pose[1][2:])])
    
    return np.array(objects), np.array(robots), np.array(r), angle 


def calculate_yaw(orientation):
    # calculate yaw angle (around x-axis)
    
    q_x = float(orientation[0][2:])
    q_y = float(orientation[1][2:])
    q_z = float(orientation[2][2:])
    q_w = float(orientation[3][2:])

    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)

    return np.arctan2(siny_cosp, cosy_cosp)

    
def transform_lidar(df, r, angle, idx, shift=0, ld=None):  
    # transform lidar data from angle - distance to point 
    
    # get lidar data 
    if not ld:
        ld = [float(el) for el in df['scan'][idx].split('ranges=[')[1].split('],')[0].split(',')]
        
    # transform lidar data to x, y positions 
        
    lidar = np.empty(shape=(360,2))

    for i in range(0, 360): 
        if not math.isinf(ld[i]):
            lidar[i][0] = ld[i] * math.cos(angle + np.deg2rad(i) + shift) + r[0]
            lidar[i][1] = ld[i] * math.sin(angle + np.deg2rad(i) + shift) + r[1]
        else: 
            lidar[i][0] = float('inf')
            lidar[i][1] = float('inf')
    
    return lidar, ld


# determine if lidar points belong to an object or kin based on gt position data

def detect_kin(robots, lidar):
    sel = np.zeros(len(lidar))
    for kin in range(len(robots)):
        robot = robots[kin]
        sel_robot = ((lidar - robot) ** 2).sum(-1) < 0.13 ** 2
        sel[sel_robot] = 1
    sel = sel.astype(bool)
    return lidar[~sel], lidar[sel], sel 


def detect_false_rotation(objects, arena, robots, r_self, lidar):
    match = np.zeros(len(lidar))
    match[np.isinf(lidar[:, 0])] = 1
    
    for kin in range(len(robots)):
        rob = robots[kin]
        # print(lidar.shape, rob.shape, r_self.shape, robots.shape)
        if ((r_self - rob) ** 2).sum(-1) > 16:
            continue
        match_kin = ((lidar - rob) ** 2).sum(-1) < 0.13 ** 2
        match[match_kin] = 1
    
    boundaries = []
    if arena == 'arena':
        boundaries = [(0, -4), (0, 4), (-4, 0), (4, 0)]
    elif arena == 'arena-large':
        boundaries = [(0, -9), (0, 9), (-9, 0), (9, 0)]
    elif arena == "two-rooms":
        boundaries = [(0, -7), (0, 7), (-9, 0), (9, 0)]
    elif arena == "arena-corners":
        boundaries = [(0, -5), (0, 5), (-4, 0), (4, 0)]
    elif arena == "arena-walls":
        boundaries = [(0, -7), (0, 7), (-7, 0), (7, 0)]
    elif arena == "arena-pillars":
        boundaries = [(0, -9), (0, 9), (-9, 0), (9, 0)]
    elif arena == "arena-poles":
        boundaries = [(0, -6), (0, 6), (-6, 0), (6, 0)]   #ToDo: other arenas and other objects 
    elif arena == "arena-walls-poles":
        boundaries = [(0, -6), (0, 6), (-6, 0), (6, 0)]
    elif arena == "arena-pillars-poles":
        boundaries = [(0, -3), (0, 3), (-6, 0), (6, 0)]
    elif arena == "arena-poles":
        boundaries = [(0, -6), (0, 6), (-6, 0), (6, 0)]
    elif arena == "arena-corners-pillars":
        boundaries = [(0, -6), (0, 6), (-4, 0), (4, 0)]
    elif arena == "turtle":
        pass # need to define objects
    elif arena == "arena-boxes-pillars":
        boundaries = [(0, -4.5), (0, 4.5), (-6, 0), (6, 0)]
    else:
       rais NotImplementedError(f"Arena {arena} is not defined.")
    
    circle_objects = []
    rectangle_objects = []
    if arena == 'arena':
        pass
    elif arena == 'arena-large':
        pass    
    elif arena == "two-rooms":
        rectangle_objects = [(-2, 2, 1, 0, 0, 6), (-2, -8, 1, 0, 0, 6)]
    elif arena == "arena-corners":
        rectangle_objects = [(-4,-3, 1, 0, 0, 1), (0, 4, 1, 0, 0, 1), (-4, 1.5, 1, 0, 0, 1.5), (3, 0, 1, 0, 0, 2), (2, -5, 2, 0, 0, 2)]
    elif arena == "arena-walls":
        rectangle_objects = [(-0.125, 2, 0.25, 0, 0, 4), (2, -1, 4, 0, 0, 0.25), (-2, -4, 0.25, 0, 0, 3)]
    elif arena == "arena-pillars":
        circle_objects = [(-4.5, 4.5, 1), (-4.5, -4.5, 1), (4.5, -4.5, 1), (4.5, 4.5, 1)]
    elif arena == "arena-poles":
        circle_objects = [(-2, -1, 0.05), (-2, -2, 0.05), (-2, -3, 0.05), (1, 4, 0.05), (3, 0, 0.05), (2, -4, 0.05), (5.9, 5.9, 0.05), (-4, 4, 0.05), (-3, 4, 0.05), (-3, 3.5, 0.05), (-4, 3.5, 0.05)]
    elif arena == "arena-walls-poles":
        rectangle_objects = [(-6, -1.125, 3, 0, 0, 0.25), (0.875, 2, 0.25, 0, 0, 4), (5.875, -6, math.sqrt(0.03125), math.sqrt(0.03125), -math.sqrt(18), -math.sqrt(18))]
        circle_objects = [(4, 1, 0.05), (5, 2, 0.05), (-1, -4, 0.05), (-2, -4, 0.05), (-2, 4, 0.05), (-3, 3, 0.05), (-4, 2, 0.05)]
    elif arena == "arena-pillars-poles":
        circle_objects = [(-4, 1, 0.4), (-1, -2, 1.2), (1, 2, 0.3), (4, -1, 0.75), (5.5, 2.5, 0.5)]
    elif arena == "arena-corners-pillars":
        rectangle_objects = [(-4, 1, 1, 0, 0, 2), (1, -6, 1, 0, 0, 1), (2, -6, 1, 0, 0, 2), (3, -6, 1, 0, 0, 3)]
        circle_objects = [(-2, -3, 0.75), (2, 4, 0.75)]
    elif arena == "turtle":
        circle_objects = [(-1.1, -1.1, 0.15), (-1.1, 0, 0.15), (-1.1, 1.1, 0.15), (0, -1.1, 0.15), (0, 0, 0.15), (0, 1.1, 0.15), (1.1, -1.1, 0.15), (1.1, 0, 0.15), (1.1, 1.1, 0.15)]
    elif arena == "arena-boxes-pillars":
        rectangle_objects = [(-4, 2, 1, 0, 0, 0.5), (-4, 0.5, 1, 0, 0, 0.5), (-4, -1, 1, 0, 0, 0.5), (-4, -2.5, 1, 0, 0, 0.5), (2, 2, 1, 0, 0, 0.5), (2, 0.5, 1, 0, 0, 0.5), (2, -1, 1, 0, 0, 0.5), (2, -2.5, 1, 0, 0, 0.5), (4.25, -1, 0.75, 0, 0, 2)]
        circle_objects = [(-0.5, 2.0, 0.7), (-0.5, -2.0, 0.7)]    
        
    for bn in boundaries:
        bx, by = bn
        if bx == 0:
            match_bn = (lidar[:, 1] - by) ** 2 < 0.15 ** 2
        if by == 0:
            match_bn = (lidar[:, 0] - bx) ** 2 < 0.15 ** 2
        match[match_bn] = 1
    
    for rec in rectangle_objects:
        x_st, y_st, x_1, y_1, x_2, y_2 = rec
        for line in [(x_st, y_st, x_1, y_1),
                    (x_st, y_st, x_2, y_2),
                    (x_st + x_2, y_st + y_2, x_1, y_1),
                    (x_st + x_1, y_st + y_1, x_2, y_2)]:
            start_pt = np.ndarray(shape=(1,2), buffer=np.array(line[:2], dtype=float), dtype=float)
            lidar_sh = lidar - start_pt
            if (abs(line[2]) > abs(line[3])):
                vec = (line[2], line[3])
                x_lid = lidar_sh[:, 0]
                y_lid = lidar_sh[:, 1]
            else:
                vec = (line[3], line[2])
                x_lid = lidar_sh[:, 1]
                y_lid = lidar_sh[:, 0]
            
            slope = vec[1] / vec[0]
            match_rec1 = (x_lid * slope - y_lid) ** 2 < 0.15 ** 2 
            match_rec2 = (x_lid ** 2 - vec[0] ** 2) < 0
            match_rec3 = x_lid * vec[0] > 0
            match_rec = match_rec1.astype(int) * match_rec2.astype(int) * match_rec3.astype(int)
            match[match_rec.astype(bool)] = 1
            
    for x, y, r in circle_objects:
        match_circ1 = ((lidar - np.ndarray(shape=(1,2), buffer=np.array([x, y], dtype=float), dtype=float)) ** 2).sum(-1) < (r + 0.15) ** 2
        match_circ2 = ((lidar - np.ndarray(shape=(1,2), buffer=np.array([x, y], dtype=float), dtype=float)) ** 2).sum(-1) > (r - 0.15) ** 2
        match_circ = match_circ1.astype(int) * match_circ2.astype(int)
        match[match_circ.astype(bool)] = 1
        
        
    prcntg = match.mean()
    chk = prcntg < 0.999
    match = match.astype(bool)
        
    return chk, lidar[~match], lidar[match], prcntg


old_cwd = os.getcwd()
os.chdir(old_cwd)
os.getcwd()


#rotations&dataset csv:

def generate_concentric_circles(center, num_circles, radius_increment):
    circles = []
    for i in range(num_circles):
        radius = i * radius_increment
        theta = np.linspace(0, 2 * np.pi, 100)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        circles.append((x, y))
    return circles

os.chdir(old_cwd)

# get folders with rosbags / lidar data  
folders = [f.name for f in os.scandir('./raw_data') if f.is_dir() and f.name not in ['X-tra runs']]

directories_dic = {}
cwd = os.getcwd()


parser = argparse.ArgumentParser()
parser.add_argument(
    '--arena', 
    type = str, 
    help = 'name of the arena')
parser.add_argument(
    '--robot_type', 
    type = str, 
    help = 'name of the robot type (burger/waffle-pi)')
parser.add_argument(
    '--nr_robots', 
    type = int, 
    help = 'number of robots')
parser.add_argument(
    '--behav', 
    type = str, 
    help = 'behaviour')
    
args = parser.parse_args()


os.chdir('./raw_data/' + args.arena + '/' + args.robot_type)

folders = os.listdir('.')

folder = [x for x in folders if x.startswith(args.arena+'_'+str(args.nr_robots)+'_'+args.robot_type+'_'+args.behav)]
folder=str(folder[0])
os.chdir('./'+folder)

files_all = os.listdir('.')
files = list(filter(lambda f: f.endswith('.csv'), files_all))
    
arena = args.arena
print(folder)
pprint(files)

#empty dataframe for rotations CSVs:
dfs = []
    
#empty dataframe for dataset CSVs:
data = {'timestamp': [],
        'robot_ID': [],
        'X': [],
        'Y':[],
        'orientation':[],
        'Flag invalid data':[],
        'Percentage': []}
for n in range(1,361):
    data['ld'+str(n)] = []
for n in range(1,361):
    data['l'+str(n)] = []
dataset = pd.DataFrame(data)
            
for rob, f in enumerate(files):
    df = pd.read_csv(f)
    for idx in range(len(df))[5:]:
        #print('next step')
        shift=[0,0,0,0,0,0,0,0,0]
        adj_1=[0,0,0,0,0,0,0,0,0]
        adj_2=[0,0,0,0,0,0,0,0,0]
    
        # get robot and object positions 
        objects, robots, r, angle = get_positions(df, idx) 
    
        # transform lidar scans to points 
        lidar, ld = transform_lidar(df, r, angle, idx, shift[rob])
    
        # check for false rotation calculation
        rot_check, lm, lnm, best_prcntg = detect_false_rotation(objects, arena, robots, r, lidar)
        best_lm, best_lnm, best_shift, best_lidar, best_adj_1, best_adj_2, = lm, lnm, shift[rob], lidar, adj_1[rob], adj_2[rob]                        
        prcntg = best_prcntg  

        #concentric circles
        # Set parameters
        center_coordinates = (0, 0)
        num_circles = 5
        radius_increment = 0.04

        # Generate concentric circles and access coordinates
        concentric_circles = generate_concentric_circles(center_coordinates, num_circles, radius_increment)

                
        for i in range(1, num_circles):
            # Accessing 50 evenly distributed points on the first circle
            num_points = 50
            circle_coordinates = concentric_circles[i]
            x_coordinates, y_coordinates = circle_coordinates

            # Extract 50 evenly distributed indices
            indices = np.linspace(0, len(x_coordinates) - 1, num_points, dtype=int)

            # Access the points using the indices
            selected_x = x_coordinates[indices]
            selected_y = y_coordinates[indices]

                    
            itr=-1
            while rot_check:
                itr+=1
                adj_1[rob]=selected_x[itr]
                adj_2[rob]=selected_y[itr]
                print(itr, end='\r')
                lidar=lidar+[adj_1[rob], adj_2[rob]]
                        
                rot_check, lm, lnm, prcntg = detect_false_rotation(objects, arena, robots, r, lidar)
                        
                if prcntg > best_prcntg:
                    best_prcntg, best_lm, best_lnm, best_adj_1, best_adj_2, best_lidar = prcntg, lm, lnm, adj_1[rob], adj_2[rob], lidar
                    #print('2. best prcntg: ' + str(best_prcntg))
                    #print('2. best adj1: ' + str(best_adj_1))
                        
                shift=[0,0,0,0,0,0,0,0,0]
                itr_2 = 0
                while rot_check:
                    itr_2 += 1
                    print(itr, end='\r')
                    shift[rob] += math.pi / 720
                    lidar, _ = transform_lidar(df, r, angle, idx, shift[rob], ld)
                    rot_check, lm, lnm, prcntg = detect_false_rotation(objects, arena, robots, r, lidar)
                    if prcntg > best_prcntg:
                        best_prcntg, best_lm, best_lnm, best_shift, best_lidar = prcntg, lm, lnm, shift[rob], lidar
                        #print('2. best prcntg: ' + str(best_prcntg))
                        #print('2. best shift: ' + str(best_shift))
                    if shift[rob] > math.pi * 2:
                        #print(folder, subfolder, rob, 'No correction found!')
                        prcntg, lm, lnm, shift[rob], lidar = best_prcntg, best_lm, best_lnm, best_shift, best_lidar
                        #print('3. prcntg: ' + str(prcntg))
                        #print('3. shift: ' + str(shift[rob]))
                        break
                        
                if itr == (num_points-1):
                    #print(folder, subfolder, rob, 'No correction found!')
                    prcntg, lm, lnm, adj_1[rob],adj_2[rob], lidar = best_prcntg, best_lm, best_lnm, best_adj_1, best_adj_2, best_lidar
                    #print('3. prcntg: ' + str(prcntg))
                    #print('3. adj1: ' + str(best_adj_1))
                    break
                        
                #print('4. prcntg: ' + str(prcntg))
                #print('4. shift: ' + str(shift[rob]))
                #print('4. rot check: ' + str(rot_check))

        # determine if object or kin
        l = detect_kin(robots, lidar)
                            
        #fillling dataframe for rotations:
        df['robot_ID'] = df['robot'].str.strip().str[-1]
        df['rotation'] = shift[rob]
        df['adj1'] = adj_1[rob]
        df['adj2'] = adj_2[rob]
                    
        #filling the dataframe for dataset CSVs:
        new_row={'timestamp': df['timestamp'][idx], 'robot_ID': rob, 'X': r[0], 'Y': r[1], 'orientation': angle + shift[rob], 'Flag invalid data': rot_check, 'Percentage': prcntg}
        for n in range(1,361):
            new_row['ld'+str(n)] = ld[n-1]
        for n in range(1,361):
            new_row['l'+str(n)] = l[2][n-1]
        dataset.loc[len(dataset)] = new_row
                    
    dfs.append(df)
                
data_rot = pd.concat(dfs)
data_ds = dataset
name = ""
name = name.join(files[0])
name = name.split("/")[0]
att= name.split("_")
if att[2]=="burger":
    date = att[8]
    filename_rot="rotations"+"_"+att[0]+"_"+att[1]+"_"+att[2]+"_"+att[3]+"_"+date+".csv"
    filename_ds="dataset"+"_"+att[0]+"_"+att[1]+"_"+att[2]+"_"+att[3]+"_"+date+".csv"
else:
    date = att[9]
    filename_rot="rotations"+"_"+att[0]+"_"+att[1]+"_"+att[2]+"_"+att[3]+"_"+att[4]+"_"+date+".csv"
    filename_ds="dataset"+"_"+att[0]+"_"+att[1]+"_"+att[2]+"_"+att[3]+"_"+att[4]+"_"+date+".csv"
path=os.getcwd()
path=path.split("/", 5)[-1]
save_to="~/arena_res/"+ str(path) +"/"
isExist = os.path.exists(save_to)
if not isExist:
    os.makedirs(save_to)
data_rot.to_csv(save_to+filename_rot,columns=['timestamp', 'robot_ID', 'rotation'], index=False) 
data_ds.to_csv(save_to+filename_ds, index=False)