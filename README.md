### Project topic: Classification of types of sleep apnea based on audio-recordings of breathing sounds

The aim of the portfolio project is to use the [PSG-Audio dataset](https://www.nature.com/articles/s41597-021-00977-w), which is a freely available open source dataset comprising 212 polysomnograms along with synchronized high-quality tracheal and ambient microphone recordings.

>*The sleep apnea syndrome is a chronic condition that affects the quality of life and increases the risk of severe health conditions such as cardiovascular diseases. However, the prevalence of the syndrome in the general population is considered to be heavily underestimated due to the restricted number of people seeking diagnosis, with the leading cause for this being the inconvenience of the current reference standard for apnea diagnosis: Polysomnography. To enhance patients’ awareness of the syndrome, a great endeavour is conducted in the literature. Various home-based apnea detection systems are being developed, profiting from information in a restricted set of polysomnography signals. In particular, breathing sound has been proven highly effective in detecting apneic events during sleep.*

A convolutional neural network architecture will be used to classify the spectrogram files of different apnea types (obstructive apnea, hypopnea, mixed apnea, central apnea) and non-apnea events based on ambient recordings alone to enable at-home analysis of sleep apnea using smartphone microphone recordings.

While many research papers have used this dataset to accurately classify apnea events, very few have done so using only ambient audio.

Challenges in this project are data processing due to the size of the dataset as well as data cleaning and feature extraction. Ambient recordings contain more noise that presumably make it more difficult to make correct classifications.

Ideally, if time allows, I would like to quantize the final model if necessary to allow it to run on a mobile device to fulfill the initial use case.