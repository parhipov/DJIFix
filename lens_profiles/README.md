# Lens profiles

Gyroflow lens profiles used with `--lens` when the camera carries a lens other
than the stock one described in the clip's telemetry. Taken from
https://github.com/gyroflow/lens_profiles (CC0 1.0). Any other profile from that
database works the same way; drop the JSON here or point `--lens` at the file.

| file | camera / lens |
|---|---|
| `DJI_O4_Air_Flywoo_O4_Wide_lens_normal_4k_4by3_3840x2880.json` | DJI O4 (Lite) air unit with the Flywoo O4 Wide lens, 4:3 4K |

Usage — the argument is a file path or any part of a file name in this folder:

```bat
fix_telemetry.bat "F:\36\clip.MP4" --lens flywoo
python src\main.py "F:\36\clip.MP4" --lens lens_profiles\DJI_O4_Air_Flywoo_O4_Wide_lens_normal_4k_4by3_3840x2880.json
```

The profile only affects how the image is lifted to rays for the pitch/yaw
measurement. Roll and the telemetry spike repair do not depend on it.
