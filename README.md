# balasa

Procedural short films in the painted style of Sabin Bălașa ("cosmic romanticism"),
drawn frame by frame with Python (numpy + Pillow) and encoded with ffmpeg. No AI video model.

## Films

| Film | Length | Code | Render |
|---|---|---|---|
| Style test | 20 s | `render.py`, `audio.py` | `renders/cosmic_romanticism_test.mp4` |
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
