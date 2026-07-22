### Candidates (max 10)

| Title | Venue/Year | Link | What it does | Why close | Key difference from the claim |
|---|---|---|---|---|---|
| 4D Attention: Comprehensive Framework for Spatio-Temporal Gaze Mapping | IEEE RA-L/2021 | [Paper](https://arxiv.org/abs/2107.03606) | Localizes eye-tracking glasses at 6-DoF in a prebuilt dense 3D map, intersects gaze with the map, and uses ID textures to distinguish surfaces and object instances. | Essentially demonstrates atomic claim (1). | Attention-analysis only: no spoken reference, robot, navigation, or manipulation; the map is not 3DGS. |
| Take That for Me: Multimodal Exophora Resolution with Interactive Questioning for Ambiguous Out-of-View Instructions (MIEL) | IEEE RO-MAN/2025 | [Paper](https://arxiv.org/abs/2508.16143) · [Project](https://emergentsystemlabstudent.github.io/MIEL/) | Resolves “Bring me that cup”-type instructions using a previously built semantic map, speech, skeletal pointing, sound-source localization, and clarification. | Strong evidence for atomic claim (2): map-based reference resolution beyond the robot’s current image. | Uses skeletal pointing rather than wearable gaze and stops at referent identification; the quantitative visibility split is principally whether the user is initially visible. |
| FAM-HRI: Foundation-Model Assisted Multimodal HRI Combining Gaze and Speech | IEEE T-ASE/2026 | [Paper](https://arxiv.org/abs/2503.16492) · [DOI](https://doi.org/10.1109/TASE.2026.3695438) | Meta Aria gaze identifies objects while speech specifies multi-step manipulation; feature matching transfers the target into a Franka’s camera view. | Direct demonstration of atomic claim (3), including commands such as selecting one fruit, plate, and cup. | Requires correspondence with the robot’s current tabletop image; no persistent map, mobile base, or remote referent. |
| Glance-Say: Multimodal HRC and Intent Recognition via Sticky Glance | arXiv/2026 | [Paper](https://arxiv.org/abs/2603.06121) | Explicitly introduces a paradigm in which “gaze specifies objects and speech specifies actions,” followed by real manipulation. | The cleanest semantic match to atomic claim (3). | Its object representation and execution are confined to a fixed-arm tabletop workspace; no global map or navigation. |
| A Multimodal Agentic AI Framework for Intuitive HRC (iBotAssistant) | Sensors/2026 | [Article](https://www.mdpi.com/1424-8220/26/6/1958) | Pupil Core gaze and speech select AprilTag-labelled lumber; a Husky–UR5e navigates from a work zone to storage, retrieves the requested piece, and returns. | Closest overall end-to-end threat: wearable gaze, speech, autonomous navigation, grasp, return, and a pre-created navigation map. | Gaze yields a 2D tag ID, not a world coordinate from a persistent 3D instance map. The experiment is one lab divided into two zones, not different rooms, and target non-visibility to the robot is not established. |
| EgoSpot: Egocentric Multimodal Control for Hands-Free Mobile Manipulation | arXiv/2023, rev. 2026 | [Paper](https://arxiv.org/abs/2306.02393) · [Project](https://ganlinzhang.xyz/Holo-Spot-Page/) | HoloLens gaze intersects its spatial mesh; Azure Spatial Anchors co-localize the point with a Spot robot, while voice and head motion control locomotion and arm functions. | Very close hardware and shared-coordinate geometry: wearable gaze, voice, quadruped, and arm. | Surface-point selection rather than persistent semantic-instance grounding; separate teleoperation modes rather than deictic action–referent fusion; no autonomous fetch. |
| HoloSpot: Intuitive Object Manipulation via Mixed Reality Drag-and-Drop | arXiv/2024 | [Paper](https://arxiv.org/abs/2410.11110) · [Project](https://holospot.github.io/) | Segments a pre-scanned 3D room with OpenMask3D; a shared object index and coordinates drive Spot navigation, grasping, carrying, placement, and return. | Strongest map-to-quadruped-manipulation counterpart. | Uses hand drag-and-drop, not gaze or spoken deixis; experiments cover a single room and do not prove a never-robot-visible target. |
| MORE: Mobile Manipulation Rearrangement Through Grounded Language Reasoning | IEEE/RSJ IROS/2025 | [Paper](https://arxiv.org/abs/2505.03035) · [Project](https://more-model.cs.uni-freiburg.de/) | Uses an object/room scene graph and metric navigation map in a real three-room apartment; an HSR explored all rooms, found and grasped a book, returned to the dining room, and placed it. | Direct real-world precedent for atomic claim (4): cross-room, map-mediated mobile fetch. | Named-language goal, robot perception, and conventional semantic mapping; no wearer, gaze, deixis, or 3DGS. |
| DynaMem: Online Dynamic Spatio-Semantic Memory for Open-World Mobile Manipulation | arXiv/2024, rev. 2025 | [Paper](https://arxiv.org/abs/2411.04999) · [Project](https://dynamem.github.io/) | Stores queryable world-coordinate voxels in a persistent 3D memory and uses them for Stretch navigation and pick-and-drop, including trials in a two-room apartment. | Strong confirmation that remembered 3D coordinates can drive home-scale manipulation. | Robot-built, language-queried memory; no gaze or deixis. The paper does not separately report which two-room trials crossed the doorway. |
| HAMMER: Heterogeneous, Multi-Robot Semantic Gaussian Splatting | IEEE RA-L/2025 | [Paper](https://arxiv.org/abs/2501.14147) · [Project](https://hammer-project.github.io/) | Aligns Aria glasses and ground robots into a global metric-semantic 3DGS map and uses semantic Gaussian queries for navigation goals. | Establishes almost exactly the proposed shared global 3DGS substrate, including wearable devices and robots. | Does not use the Aria gaze or speech; semantics are CLIP fields rather than persistent object-instance selection, and there is no manipulation or fetch. |

The four atomic answers are therefore:

1. **Wearable gaze → world-frame object grounding in a persistent map: YES.** 4D Attention directly performs 6-DoF wearable localization and gaze-to-map/object association. HAMMER separately establishes that wearables and robots can share a global metric-semantic 3DGS, but nobody found combines those two exact implementations.

2. **Deixis with the referent outside the robot’s view: YES.** [RO-MAN 2023 exophora resolution](https://doi.org/10.1109/RO-MAN57019.2023.10309487) and MIEL use previously mapped object information to consider referents absent from the robot’s current image. They use pointing/context rather than wearable gaze and do not execute retrieval.

3. **Speech supplies action + gaze supplies referent: YES.** Glance-Say states this division verbatim; FAM-HRI and iBotAssistant independently demonstrate it with physical robot execution.

4. **Cross-room mobile manipulation fetch from map coordinates: YES.** MORE demonstrates a three-room fetch-and-return using a semantic/navigation map. DynaMem and BUMBLE provide additional persistent-coordinate and building-scale precedents, respectively.

**Single-system three-or-more result:** No verified system satisfies three of these four atoms *as written*. iBotAssistant would appear to reach three only if “global 3D map grounding” were weakened to “selecting an AprilTag ID” and “cross-room” were weakened to “cross-zone within one lab.”

### Closest 3

1. [iBotAssistant](https://www.mdpi.com/1424-8220/26/6/1958)

For pillar (a), it genuinely combines a wearable eye tracker with speech: gaze identifies a particular tagged lumber piece and speech provides requests such as retrieval or holding. For pillar (b), however, gaze produces an AprilTag ID from the wearer’s current image; it does not transform a localized gaze ray into a persistent global 3D object map, and the paper does not establish that the target was absent from every robot frame before commitment. For pillar (c), it performs autonomous navigation, pickup, return, and delivery, but only between two zones of one lab. This is the principal citation that makes a broad “first gaze–speech mobile fetch” claim untenable.

2. [Glance-Say](https://arxiv.org/abs/2603.06121)

Pillar (a) is essentially exact: the paper explicitly defines gaze as selecting the object and speech as selecting the action. Pillar (b) is absent because grounding occurs in a locally explored tabletop representation, not a persistent multi-room world map, and the target is within the shared manipulation workspace. Pillar (c) is also absent because execution uses a stationary arm. It establishes that the interaction semantics themselves cannot be claimed as novel.

3. [HoloSpot](https://arxiv.org/abs/2410.11110)

Pillar (a) is missing: object selection is hand-based drag-and-drop, while voice only exposes interface functions such as showing or resetting objects. Pillar (b) is strong but incomplete: a pre-scanned, instance-segmented 3D representation supplies object identity and world coordinates, although only for one room and without a controlled never-visible condition. Pillar (c) is also strong: Spot navigates to the stored instance, grasps it, transports it, places it, and returns. It shows that the map-to-mobile-manipulator half of the composition is already mature.

### Verdict

**CLAIM SURVIVES.** As of July 19, 2026, I found no verified system demonstrating wearable-gaze deixis resolved through a persistent global metric instance map while the referent is unavailable to the robot, followed by different-room mobile retrieval. The defensible formulation is: “To our knowledge, this is the first end-to-end real-robot demonstration in which world-registered wearable gaze resolves a spoken deictic reference to an object instance in a persistent global metric map before that instance has been observed by the robot during reference resolution, and a mobile manipulator retrieves it from another room.”

### Search log

Representative queries:

- `"wearable gaze" world coordinates 3D map mobile eye tracker gaze ray semantic map`
- `"4D Attention" gaze mapping pre-built 3D map mobile eye tracker`
- `"SLAM-based Localization of 3D Gaze" world coordinates`
- `"Take That for Me" out-of-view robot MIEL`
- `robot deixis resolution referent outside robot field of view semantic map "that cup"`
- `ECRAP exophora resolution demonstrative robot semantic map`
- `"FAM-HRI" gaze speech Meta Aria object task robot`
- `"Glance-Say" gaze speech robot action referent`
- `wearable eye gaze supplies object speech supplies action robot manipulation`
- `mobile manipulator fetch object across rooms semantic map coordinate`
- `DynaMem two-room home mobile manipulation memory coordinates`
- `BUMBLE building-wide mobile manipulation map`
- `MORE three rooms fetch book semantic map`
- `"iBotAssistant" gaze speech Husky UR5e retrieve`
- `HoloSpot OpenMask3D Spot grasp carry place`
- `EgoSpot HoloLens gaze Azure Spatial Anchors Spot arm voice`
- `HAMMER semantic Gaussian splatting Aria global map`
- `("eye gaze" OR "wearable gaze") "Gaussian Splatting" robot manipulation speech`
- `site:scholar.google.com "wearable gaze" "mobile manipulation" speech`
- `site:youtube.com wearable gaze speech mobile robot fetch object demonstration`

There was no general web-search outage. The Google Scholar- and YouTube-restricted queries returned no useful indexed candidate pages; embedded videos on authors’ project pages were used instead. A few direct PDF/publisher opens returned internal errors, but their arXiv, DOI, or project-page counterparts remained accessible.

## Composition verdict

Yes—the composition is a defensible **systems-demonstration contribution**, provided none of the four atoms is presented as individually novel. Cross-room manipulation, gaze–speech fusion, global 3DGS mapping, and map-based out-of-view reference resolution all have clear prior art; the contribution is the experimentally demonstrated bridge among them.

The weakest link is proving that the persistent map—not a tag, live robot perception, or later reacquisition—actually resolves the referent. Define “outside the robot’s view” as no target pixels in any robot sensor stream from command onset until the target instance/world coordinate is committed; after navigation, local vision for grasping is necessarily allowed. Duplicate-cup trials, gaze/pose/map ablations, and end-to-end error reporting are essential, because iBotAssistant already makes the broader “wearable gaze + speech + mobile delivery” composition non-novel.