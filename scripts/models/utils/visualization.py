import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from models.utils import graph

# find path to ffmpeg via terminal
# which ffmpeg
plt.rcParams['animation.ffmpeg_path'] =  '/usr/bin/ffmpeg' #'/usr/local/ffmpeg/3.4.2/bin/ffmpeg'
def display_video(video):
    fig = plt.figure(figsize=(3,3))  #Display size specification

    mov = []
    for i in range(len(video)):  #Append videos one by one to mov
        img = plt.imshow(video[i], animated=True)
        plt.axis('off')
        mov.append([img])

    #Animation creation
    anime = animation.ArtistAnimation(fig, mov, interval=50, repeat_delay=1000)

    plt.close()
    return anime

# def viz_pose(data):
#     # Data preprocessing
#     data[data[:, :, :, -1] == 0.5] = 0.0
#     data[data[:, :, :, -1] == -0.5] = 0.0

#     C, T, V, M = data.shape

#     # Define the connections in your skeleton
#     coco_inward = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5),
#                 (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0),
#                 (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
#     edge = coco_inward

#     # Prepare the figure
#     fig, ax = plt.subplots(figsize=(3,3))
#     ax.grid(True)
#     ax.axis([-1, 1, -1, 1])

#     # This function is called to update the plot for each frame
#     def update_graph(t):
#         ax.clear()  # Clear previous frame
#         ax.grid(True)
#         ax.axis([-1, 1, -1, 1])
#         for m in range(M):
#             for v1, v2 in edge:
#                 ax.plot(data[0, t, [v1, v2], m], -data[1, t, [v1, v2], m], 'b-')
#         return ax

#     # Create the animation
#     ani = animation.FuncAnimation(fig, update_graph, frames=T, interval=50)

#     plt.close()  # Prevents the static plot from showing
#     return ani

def viz_pose(data, face, r_hand, l_hand):
    C, T, V, M = data.shape

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(3,3))
    ax.grid(True)
    ax.axis([-1, 1, -1, 1])

    # This function is called to update the plot for each frame
    def update_graph(t):
        ax.clear()  # Clear previous frame
        ax.grid(True)
        ax.axis([-1, 1, -1, 1])
        for m in range(M):
            for v1, v2 in graph.coco_inward_edges:
                ax.plot(data[0, t, [v1, v2], m], -data[1, t, [v1, v2], m], 'b-')
            for v1, v2 in graph.hand_inward_edges:
                ax.plot(r_hand[0, t, [v1, v2], m], -r_hand[1, t, [v1, v2], m], 'r-')
            for v1, v2 in graph.hand_inward_edges:
                ax.plot(l_hand[0, t, [v1, v2], m], -l_hand[1, t, [v1, v2], m], 'g-')
            for v1, v2 in graph.face_inward_edges:
                ax.plot(face[0, t, [v1, v2], m], -face[1, t, [v1, v2], m], 'c-')
        return ax

    # Create the animation
    ani = animation.FuncAnimation(fig, update_graph, frames=T, interval=50)

    plt.close()  # Prevents the static plot from showing
    return ani