# balasa

Procedural short films in the painted style of Sabin Bălașa ("cosmic romanticism"),
drawn frame by frame with Python (numpy + Pillow) and encoded with ffmpeg. No AI video model.

## Films

| Film | Length | Code | Render |
|---|---|---|---|
| Style test | 20 s | `render.py`, `audio.py` | `renders/cosmic_romanticism_test.mp4` |
| The Drowned Temple | 56 s | `temple.py` | `renders/the_drowned_temple.mp4` (silent) |
| One Dot | 56 s | `rd_sim.py` then `onedot.py` | `renders/one_dot.mp4` (silent) |
| Soft Geology | 56 s | `soft.py`, `relief.py` | `renders/soft_geology.mp4` (silent) |
| Stone, Moth, Horse, Rain | 56 s | `metamorph.py`, `cutouts.py` | `renders/stone_moth_horse_rain.mp4` (silent) |
| Birds and Thread | 52 s | `birds.py` | `renders/birds_and_thread.mp4` (silent, stop-motion) |
| The Place Between Dreams | 55 s | `dreams.py`, `dreams_audio.py` | `renders/the_place_between_dreams.mp4` |
| The Hotel That Dreams of Me | 60 s | `film.py`, `characters.py` (`film_audio.py` optional) | `renders/the_hotel_that_dreams_of_me.mp4` (silent) |

## Render

Needs Python 3, numpy, Pillow, ffmpeg, and espeak-ng (hotel film voice only).

```bash
python3 dreams.py frames/
python3 dreams_audio.py dreams.wav
ffmpeg -framerate 24 -i frames/f_%04d.png -i dreams.wav -c:v libx264 -pix_fmt yuv420p -crf 17 -c:a aac -shortest out.mp4
```

Render specific frames to preview: `python3 dreams.py frames/ 100 400 900`.

## Layout

- `lib.py`: textures, gradients, masks, colour helpers.
- `engine.py`: shot timeline, dissolves, virtual camera, paint/film finishing, parallel render.
- `characters.py`: bellhop, faces, figures (hotel film).
- Each film file defines its shots as functions of time and a `SHOTS` list.

## Notes

- The hotel film render is silent (made for music edits). `film.py` applies a "Balasa grade" to every frame:
  darker tones, muted colour with saturated accents kept, shadows tinted midnight blue, heavier pastel hatching.
- To encode without audio: `ffmpeg -framerate 24 -i frames/f_%04d.png -an -c:v libx264 -pix_fmt yuv420p -crf 17 out.mp4`.

## Museum cut-outs

`metamorph.py` uses real public-domain / CC0 museum images, repainted into the Balasa palette by `cutouts.py`.
They live on the `fetch-assets` branch (`assets/cut`, `assets/raw`, credits in `assets/manifest.json`).
That branch's GitHub Action (`tools/fetch_assets.py`) searches the Cleveland Museum of Art, Wikimedia Commons
and Artvee, removes backgrounds, and commits the results. Edit `tools/queries.txt` and push to fetch more.
Point `BALASA_ASSETS` at the `assets/cut` folder when rendering.

## One Dot

A single Gray-Scott reaction-diffusion simulation (no cuts), steered by chemistry schedules and by image maps
(a marble head and a moth engraving) that it grows into like a halftone. Run `python3 rd_sim.py states.npy`
(about 3 minutes), then `python3 onedot.py states.npy frames/`.

## slim-e world, Balasa cut

`slime_world.py` turns one still of the slim-e homepage world film into a seamless 15 s stop-motion loop:
every friend, cloud and sun ray is cut into its own piece, sky and ground are repainted as pastel on paper,
and one cloud breaks into paper birds that circle and land back. Drop-in replacements for the site's
`media/world-d.mp4` / `world-m.mp4` (+ `.webp` posters) are in `renders/slime/` (under 1 MB each, 12 fps).

    python3 slime_world.py world_frame_d.png out_d 960 594 [rembg_cut.png]   # desktop, coin centre
    python3 slime_world.py world_frame_m.png out_m 414 842                   # mobile

## Slim-E and Friends (story)

`slime_story.py`: 50 s, 4:3, silent. Slim-E (vinyl figure photos) wakes in his grey clay world; the eight
hoodie sets fall from the clouds onto two clotheslines and bring each friend's colour back; the sets walk
off behind him; at night their star-eyed faces rise and become the stars. Render: `renders/slime/slim_e_and_friends.mp4`.
Needs the cut-outs on the `fetch-assets` branch (`assets/site_cut/`).
