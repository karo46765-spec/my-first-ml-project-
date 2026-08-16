import moviepy as mp

def convert_gif_to_mp4(C:\Users\Administrator\Downloads\0e8fb9237b6925c0e775a6b050d27e95.gif, C:\Users\Administrator\Downloads\):
    # Load the GIF file
    clip = mp.VideoFileClip(gif_path)
    
    # Write the result to a video file
    clip.write_videofile(output_path, codec="libx264")
    
    # Close the clip to release resources
    clip.close()

# Example usage
convert_gif_to_mp4("input.gif", "output.mp4")
