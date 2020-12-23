### Audio samples: https://wendison.github.io/FCL-taco2-demo/

### Code: As the work is done during internship in the company, the code needs to be verfied to contain no confidential information of company, so the code is being checked now, we will release the code once the checking process is done!


*  Step1. Data preparation & preprocessing

1.  Download LJSpeech from https://keithito.com/LJ-Speech-Dataset/

2.  Unpack downloaded LJSpeech-1.1.tar.bz2 to some place in your machine, e.g., /xx/LJSpeech-1.1

3.  Obtain the forced alignment information by using Montreal forced aligner tool https://montreal-forced-aligner.readthedocs.io/en/latest/, you can also download our alignment result at xxx, then unpack it to some palce in your machine, e.g., /xx/TextGrid

4.  Preprocess the dataset to extract mel-spectrograms, phoneme duration, pitch, energy and phoneme sequence by:

         python preprocessing.py --data-root /xx/LJSpeech-1.1 --textgrid-root /xx/TextGrid



*  Step2. Model training

1.  Training teacher model FCL-taco2-T: ./teacher_model_training.sh

2.  Training student model FCL-taco2-S: ./student_model_training.sh


*  Step3. Model evaluation

1.  Train a new vocoder or download Parallel-WaveGAN vocoder from https://drive.google.com/open?id=1a5Q2KiJfUQkVFo5Bd1IoYPVicJGnm7EL, then put it under 'vocoder'

2.  FCL-taco2-T evaluation: ./inference_teacher.sh

3.  FCL-taco2-S evaluation: ./inference_student.sh


