import numpy as np


########################################################################
#                         Download information                         #
########################################################################

# Credit: https://github.com/torch-points3d/torch-points3d

FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1"
ZIP_NAME = "Stanford3dDataset_v1.2.zip"
ALIGNED_ZIP_NAME = "Stanford3dDataset_v1.2_Aligned_Version.zip"
UNZIP_NAME = "Stanford3dDataset_v1.2"
ALIGNED_UNZIP_NAME = "Stanford3dDataset_v1.2_Aligned_Version"


########################################################################
#                              Data splits                             #
########################################################################

# Credit: https://github.com/torch-points3d/torch-points3d

ROOM_TYPES = {
    "conferenceRoom": 0,
    "copyRoom": 1,
    "hallway": 2,
    "office": 3,
    "pantry": 4,
    "WC": 5,
    "auditorium": 6,
    "storage": 7,
    "lounge": 8,
    "lobby": 9,
    "openspace": 10}

VALIDATION_ROOMS = [
    "hallway_1",
    "hallway_6",
    "hallway_11",
    "office_1",
    "office_6",
    "office_11",
    "office_16",
    "office_21",
    "office_26",
    "office_31",
    "office_36",
    "WC_2",
    "storage_1",
    "storage_5",
    "conferenceRoom_2",
    "auditorium_1"]

ROOMS = {
    "Area_1": [
        "conferenceRoom_1",
        "conferenceRoom_2",
        "copyRoom_1",
        "hallway_1",
        "hallway_2",
        "hallway_3",
        "hallway_4",
        "hallway_5",
        "hallway_6",
        "hallway_7",
        "hallway_8",
        "office_1",
        "office_10",
        "office_11",
        "office_12",
        "office_13",
        "office_14",
        "office_15",
        "office_16",
        "office_17",
        "office_18",
        "office_19",
        "office_2",
        "office_20",
        "office_21",
        "office_22",
        "office_23",
        "office_24",
        "office_25",
        "office_26",
        "office_27",
        "office_28",
        "office_29",
        "office_3",
        "office_30",
        "office_31",
        "office_4",
        "office_5",
        "office_6",
        "office_7",
        "office_8",
        "office_9",
        "pantry_1",
        "WC_1"],
    "Area_2": [
        "auditorium_1",
        "auditorium_2",
        "conferenceRoom_1",
        "hallway_1",
        "hallway_10",
        "hallway_11",
        "hallway_12",
        "hallway_2",
        "hallway_3",
        "hallway_4",
        "hallway_5",
        "hallway_6",
        "hallway_7",
        "hallway_8",
        "hallway_9",
        "office_1",
        "office_10",
        "office_11",
        "office_12",
        "office_13",
        "office_14",
        "office_2",
        "office_3",
        "office_4",
        "office_5",
        "office_6",
        "office_7",
        "office_8",
        "office_9",
        "storage_1",
        "storage_2",
        "storage_3",
        "storage_4",
        "storage_5",
        "storage_6",
        "storage_7",
        "storage_8",
        "storage_9",
        "WC_1",
        "WC_2"],
    "Area_3": [
        "conferenceRoom_1",
        "hallway_1",
        "hallway_2",
        "hallway_3",
        "hallway_4",
        "hallway_5",
        "hallway_6",
        "lounge_1",
        "lounge_2",
        "office_1",
        "office_10",
        "office_2",
        "office_3",
        "office_4",
        "office_5",
        "office_6",
        "office_7",
        "office_8",
        "office_9",
        "storage_1",
        "storage_2",
        "WC_1",
        "WC_2"],
    "Area_4": [
        "conferenceRoom_1",
        "conferenceRoom_2",
        "conferenceRoom_3",
        "hallway_1",
        "hallway_10",
        "hallway_11",
        "hallway_12",
        "hallway_13",
        "hallway_14",
        "hallway_2",
        "hallway_3",
        "hallway_4",
        "hallway_5",
        "hallway_6",
        "hallway_7",
        "hallway_8",
        "hallway_9",
        "lobby_1",
        "lobby_2",
        "office_1",
        "office_10",
        "office_11",
        "office_12",
        "office_13",
        "office_14",
        "office_15",
        "office_16",
        "office_17",
        "office_18",
        "office_19",
        "office_2",
        "office_20",
        "office_21",
        "office_22",
        "office_3",
        "office_4",
        "office_5",
        "office_6",
        "office_7",
        "office_8",
        "office_9",
        "storage_1",
        "storage_2",
        "storage_3",
        "storage_4",
        "WC_1",
        "WC_2",
        "WC_3",
        "WC_4"],
    "Area_5": [
        "conferenceRoom_1",
        "conferenceRoom_2",
        "conferenceRoom_3",
        "hallway_1",
        "hallway_10",
        "hallway_11",
        "hallway_12",
        "hallway_13",
        "hallway_14",
        "hallway_15",
        "hallway_2",
        "hallway_3",
        "hallway_4",
        "hallway_5",
        "hallway_6",
        "hallway_7",
        "hallway_8",
        "hallway_9",
        "lobby_1",
        "office_1",
        "office_10",
        "office_11",
        "office_12",
        "office_13",
        "office_14",
        "office_15",
        "office_16",
        "office_17",
        "office_18",
        "office_19",
        "office_2",
        "office_20",
        "office_21",
        "office_22",
        "office_23",
        "office_24",
        "office_25",
        "office_26",
        "office_27",
        "office_28",
        "office_29",
        "office_3",
        "office_30",
        "office_31",
        "office_32",
        "office_33",
        "office_34",
        "office_35",
        "office_36",
        "office_37",
        "office_38",
        "office_39",
        "office_4",
        "office_40",
        "office_41",
        "office_42",
        "office_5",
        "office_6",
        "office_7",
        "office_8",
        "office_9",
        "pantry_1",
        "storage_1",
        "storage_2",
        "storage_3",
        "storage_4",
        "WC_1",
        "WC_2"],
    "Area_6": [
        "conferenceRoom_1",
        "copyRoom_1",
        "hallway_1",
        "hallway_2",
        "hallway_3",
        "hallway_4",
        "hallway_5",
        "hallway_6",
        "lounge_1",
        "office_1",
        "office_10",
        "office_11",
        "office_12",
        "office_13",
        "office_14",
        "office_15",
        "office_16",
        "office_17",
        "office_18",
        "office_19",
        "office_2",
        "office_20",
        "office_21",
        "office_22",
        "office_23",
        "office_24",
        "office_25",
        "office_26",
        "office_27",
        "office_28",
        "office_29",
        "office_3",
        "office_30",
        "office_31",
        "office_32",
        "office_33",
        "office_34",
        "office_35",
        "office_36",
        "office_37",
        "office_4",
        "office_5",
        "office_6",
        "office_7",
        "office_8",
        "office_9",
        "openspace_1",
        "pantry_1"]
    }


########################################################################
#                                Labels                                #
########################################################################

# Credit: https://github.com/torch-points3d/torch-points3d

SEGMENT3D_NUM_CLASSES = 4

INV_OBJECT_LABEL = {
    0: "base",
    1: "pump",
    2: "tank",
    3: "valve"
    
    # 4: "pipe",
    # 5: "elbow",
    # 6: "column",
    # 7: "beam",
    # 8: "angled beam"

    
    }

CLASS_NAMES = [INV_OBJECT_LABEL[i] for i in range(SEGMENT3D_NUM_CLASSES)] + ['ignored']

CLASS_COLORS = np.asarray([
    [0, 0, 0],        # 'unlabelled' -> black
    [233, 229, 107],  
    [95, 156, 196],   
    [179, 116, 81] 


    # [241, 149, 131],  
    # [81, 163, 148],   
    # [77, 174, 84],    
    # [108, 135, 75],   
    # [41, 49, 101]
    
    ])      

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

def object_name_to_label(object_class):
    """Convert from object name to int label. By default, if an unknown
    object nale
    """
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["clutter"])
    return object_label

# For instance segmentation
MIN_OBJECT_SIZE = 100
STUFF_CLASSES = []
THING_CLASSES =  np.array(list(range(SEGMENT3D_NUM_CLASSES)))
STUFF_CLASSES_MODIFIED = [0, 1, 2]
THING_CLASSES_MODIFIED = list(range(3, SEGMENT3D_NUM_CLASSES))
