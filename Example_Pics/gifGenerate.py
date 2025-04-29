from PIL import Image

frames = [Image.open(f'{i}.png') for i in range(17)]

extra_pause_frames = 4
for _ in range(extra_pause_frames):
    frames.append(frames[-1])


frames[0].save(
    'Banyassady.gif',
    save_all=True,
    append_images=frames[1:],
    duration=1020,
    loop=0
)