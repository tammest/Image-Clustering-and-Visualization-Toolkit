import os
import zarr
import numpy as np
from skimage.measure import block_reduce
from skimage.transform import resize
from skimage.io import imsave

def gen_tiles(output_path: str, sample_name: str, tile_size: int = 128) -> None:
    ''' Generate tiles for a given slide '''
    slide_path = f'{output_path}/{sample_name}/data.zarr'
    tiles_output_path = f'{output_path}/{sample_name}/tiles'
    os.makedirs(tiles_output_path, exist_ok=True)
    
    # Read slide
    print('Reading slide...')
    slide = zarr.load(slide_path)
    
    # Generate and save thumbnail
    print('Generating thumbnail...')
    thumbnail = gen_thumbnail(slide, scaling_factor=tile_size // 4)
    print(f'Thumbnail shape: {thumbnail.shape}, dtype: {thumbnail.dtype}')
    save_img(tiles_output_path, 'thumbnail', tile_size // 4, thumbnail)
    
    # Generate and save mask
    print('Generating mask...')
    mask = gen_mask(thumbnail)
    print(f'Mask shape: {mask.shape}, dtype: {mask.dtype}')
    save_img(tiles_output_path, 'mask', tile_size // 4, mask)
    
    # Generate and save tile positions
    print('Generating tile positions...')
    tile_img, positions = gen_tile_positions(slide, mask, tile_size=tile_size)
    print(f'Generated {len(positions)} tiles for slide with shape {slide.shape}')
    save_img(tiles_output_path, 'tile_img', tile_size, tile_img)
    
    # Save positions to CSV
    with open(os.path.join(tiles_output_path, f'positions_{tile_size}.csv'), 'w') as f:
        f.write(' ,h,w\n')
        for i, (h, w) in enumerate(positions):
            f.write(f'{i},{h},{w}\n')
    
    print(f'Generated {len(positions)} tiles for slide with shape {slide.shape}')

def save_img(output_path: str, task: str, tile_size: int, img: np.ndarray):
    ''' Save image to output path '''
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    imsave(os.path.join(output_path, f'{task}_{tile_size}.png'), img)

def gen_thumbnail(slide: zarr, scaling_factor: int) -> np.ndarray:
    ''' Generate thumbnail for a given slide '''
    # Make sure first channel is smaller than the second and third
    assert slide.shape[0] < slide.shape[1] and slide.shape[0] < slide.shape[2]
    cache = block_reduce(slide,
                         block_size=(slide.shape[0], scaling_factor, scaling_factor),
                         func=np.mean)
    # Remove bright pixels top 5 percentile
    cache = np.clip(cache, 0, np.percentile(cache, 95))
    cache /= cache.max()
    thumbnail = np.clip(cache, 0, 1).squeeze()
    return thumbnail

def gen_mask(thumbnail: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    ''' Generate mask for a given thumbnail '''
    mask = np.where(thumbnail > threshold, 1, 0)
    mask = mask.astype(np.uint8)  # Ensure mask is binary (0 or 1)
    return mask

def gen_tile_positions(slide: zarr, mask: np.ndarray, tile_size: int = 128) -> tuple:
    ''' Generate tiles for a given slide and mask '''
    # Read numpy dimensions
    _, slide_height, slide_width = slide.shape
    grid_height, grid_width = slide_height // tile_size, slide_width // tile_size
    
    # Convert mask to pixel level grid
    mask_resized = resize(mask, (grid_height, grid_width), order=0, anti_aliasing=False)
    
    # Generate mask
    tile_img = np.where(mask_resized > 0, 1, 0)
    
    # Generate tiles
    hs, ws = np.where(mask_resized > 0)
    positions = np.array(list(zip(hs, ws))) * tile_size
    
    return tile_img, positions

# Example usage
output_path = '/gpfs/scratch/tm3475'
sample_name = '20230513_KPM-PB-20_Scan1T.ome'
gen_tiles(output_path, sample_name)
