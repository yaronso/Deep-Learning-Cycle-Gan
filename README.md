# Deep-Learning-Cycle-GAN
Deep learning project using Cycle-GAN - image to image translation.

Please follow the following installations, prior to program execution (Training/Test Notebooks):

```
!pip install tensorflow_addons
!pip install --upgrade tensorflow
!pip install --upgrade tensorflow-gpu
```

After the installaion phase, run the Training/Test Google Colab notebooks:

Links to both Training phase and Test phase notebooks:
https://colab.research.google.com/drive/1ddlkwt812uk1RHQUJ9lNO-4sHnXNdnW6?usp=sharing
https://colab.research.google.com/drive/1SSlKE5j3Oeqyz4OK6OOzoDheTg86vDwR?usp=sharing

When running both Train and Test(Inference) notebooks, use the supplied monet and photos tfrec files in order to load the data in the program. Make sure the tfrec files are in the correct path (set the path in the program).

When running the Test notebook, use the following files in order to load the model (put those files in any path on, change the PATH variable's value inside the Test notebook to your environment path):
1) saved_model.pb
2) keras_metadata.pb
3) variables.data-00000-of-00001
4) variables.index

Note: The tfrec files were added also to our git repository.

Note: All images (for the training process) are located in this repository, and saved as tfrec files (binary storage format). They are automatically loaded on program execution.
