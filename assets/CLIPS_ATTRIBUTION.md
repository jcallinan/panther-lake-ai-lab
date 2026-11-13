# Stock video clips: sources and licences

The demos use short clips from Wikimedia Commons. They are **not committed**. `examples/common/clips.py`
downloads each one on first use from Commons' server-side transcode, trims it to the segment below,
scales it to 720p or less, strips the audio, and caches it as `assets/clips/<name>.mp4` (gitignored).
We checked each licence on its Commons file page (licence template and structured metadata) on 2026-11-11.

| Name | Clip (Commons file page) | Author | Licence | Segment used | Used for |
|---|---|---|---|---|---|
| `street` | [DiagonalCrosswalkYongeDundas.webm](https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm) | Raysonho @ Open Grid Scheduler / Grid Engine | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | 4–28 s | Detection: busy Toronto crossing with pedestrians, cars and a bus |
| `person` | [Video Blog- CBP Super Bowl LII Countdown to Kickoff - Day 4.webm](https://commons.wikimedia.org/wiki/File:Video_Blog-_CBP_Super_Bowl_LII_Countdown_to_Kickoff_-_Day_4.webm) | Mani Albrecht, U.S. Customs and Border Protection Office of Public Affairs | Public domain: U.S. federal government work ([PD-USGov-DHS](https://commons.wikimedia.org/wiki/Template:PD-USGov-DHS); Commons licence review passed) | 1–25 s | Segmentation and pose: one person facing the camera, upper body visible |
| `forest` | [Fern Canyon in Prairie Creek Redwoods State Park.webm](https://commons.wikimedia.org/wiki/File:Fern_Canyon_in_Prairie_Creek_Redwoods_State_Park.webm) | Thomas Farley | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | 4–26 s (title cards removed) | Depth: a creek bed between fern walls, receding into the distance |
| `drive` | [Dashcam Recording (urban).ogv](https://commons.wikimedia.org/wiki/File:Dashcam_Recording_(urban).ogv) | Fernost | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | 40–65 s | Depth and detection: a dashcam drive through an office district |

None of these licences requires attribution. We credit the authors anyway, as a courtesy.
The `person` clip shows a U.S. government employee. Public-domain status covers copyright only. It does
not imply that the person or agency endorses this project. Do not use the clip to suggest endorsement.

Suggested credit line:

> Footage: "DiagonalCrosswalkYongeDundas" by Raysonho (CC0); "Video Blog: CBP Super Bowl LII Countdown, Day 4" by U.S. CBP (public domain);
> "Fern Canyon in Prairie Creek Redwoods State Park" by Thomas Farley (CC0); "Dashcam Recording (urban)" by Fernost (CC0). All via Wikimedia Commons.
