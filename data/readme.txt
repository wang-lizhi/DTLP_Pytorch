==============================================================================
A Few Descriptions of Datasets
==============================================================================
1. How to make datasets?
For training, trainsets and validsets are derived from the same data in a ratio of four to one. First, read the complete images from .mat files and normalize them to 0-1. 
Then, obtain the patches according to a given stride and save them in HDF5 files. Note that, each .h5 file should be in the same length and the patches should be shuffled.
For testing, the stride used for obtaining patches is equal to half of the patch size. That is, half of the adjacent patches are overlapped. 
When interfacing the model, the overlapping areas shoud be averaged. Afterwards, the stitched images are used to calculate the performance metrics.
==============================================================================
2. How to adjust settings in "utils.py"?
First, select the "mode". In "paper" mode, one mask is randomly generated for one batch. In "cave" mode, each mask is a random patch of a given real mask.
Second, modify the paths including "pretrained_path", "trainset_path", and "testset_path".
Third, set "gpuid", "batch_size", "batch_num", "train_len", and "valid_len". "Batch_num" represents the number of batches of one .h5 file. 
"Train_len" and "valid_len" represent the number of .h5 files.
Finally, check "channel" and "blocksize" according to your data. Depending on whether you need to load the pre-trained model, set "pretrained" to True or False.
==============================================================================
3. How to split datasets?
For Havrad dataset, 35 scenes are used for training, and 9 scenes are used for testing. These images are published in 
http://vision.seas.harvard.edu/hyperspec/. The split filenames are as follows:
---------Havrad_train---------
001: img1
002: img2
003: imga1
004: imga2
005: imga5
006: imga6
007: imga7
008: imgb0
009: imgb1
010: imgb2
011: imgb3
012: imgb4
013: imgb5
014: imgb6
015: imgb7
016: imgb8
017: imgb9
018: imgc1
019: imgc2
020: imgc4
021: imgc5
022: imgc7
023: imgc8
024: imgc9
025: imgd2
026: imgd3
027: imgd4
028: imgd7
029: imgd8
030: imgd9
031: imge0
032: imge1
033: imge2
034: imge3
035: imge4

---------Havard_test---------
001: imge7
002: imgf3
003: imgf4
004: imgf5
005: imgf7
006: imgh0
007: imgh1
008: imgh2
009: imgh3

For CAVE dataset, 30 scenes are used for training. These images are published in https://www1.cs.columbia.edu/CAVE/projects/gap_camera/. 
The filenames are as follows:
---------CAVE_train---------
001: balloons_ms
002: beads_ms
003: cd_ms
004: chart_and_stuffed_toy_ms
005: clay_ms
006: cloth_ms
007: egyptian_statue_ms
008: face_ms
009: beers_ms
010: lemon_slices_ms
011: lemons_ms
012: peppers_ms
013: strawberries_ms
014: sushi_ms
015: tomatoes_ms
016: feathers_ms
017: flowers_ms
018: glass_tiles_ms
019: hairs_ms
020: jelly_beans_ms
021: oil_painting_ms
022: paints_ms
023: photo_and_face_ms
024: pompoms_ms
025: yellowpeppers_ms
026: sponges_ms
027: stuffed_toys_ms
028: superballs_ms
029: thread_spools_ms
030: watercolors_ms

For KAIST dataset, 10 scenes are used for testing. These images are published in http://vclab.kaist.ac.kr/siggraphasia2017p1/. 
The filenames are as follows:
---------KAIST_test---------
001: scene03
002: scene04
003: scene05
004: scene08
005: scene11
006: scene12
007: scene14
008: scene21
009: scene23
010: scene24