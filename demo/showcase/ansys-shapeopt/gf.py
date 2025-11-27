import imageio

images = []

for i in range(10):
    filename = f"tmp_img/mesh_optim_{i:03d}.png"
    images.append(imageio.imread(filename))
    print(f"Added {filename} to gif.")
# make sure the gif repeats forever
imageio.mimsave("mesh_optim.gif", images, fps=10, loop=0)
