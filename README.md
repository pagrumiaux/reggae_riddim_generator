# Reggae Riddim Generator

This is a personal project, still in early development, which progresses during my free time.
The objective is to create a neural network based algorithm that can generate reggae riddims on-the-fly. A riddim is the name of the musical sequence which forms the basis of a reggae song. The choice of the reggae music genre is two-fold : it's a music style I love and with which I'm quite familiar, and also because it's a genre that have a recognizable pattern with small changes, so it is a good start in creating an AI-based music generation tool.

### System overview

The system will be based of several components :
* A chord progression generator, which will create the riddim's harmonic progression. It will drive other musical sequences such as the bass or other melodical instruments. It will also be the chords played by the *skank* part of the riddim.
* A bass melody generator, which will create a melody for the bass instrument, based on the chord progression previously generated.
* *(For later)* a generic melody generator. This might be interesting when dealing with modern reggae/dub music that have a lot of melodic (electronic) instruments.

The goal is to be able to generate a whole skank+bass riddim (with generic drums pattern, as it is very often the same pattern in reggae music) by starting with a prompted/randomized starting chord.

### Dataset

To train the AI-based algorithm, we need some data. That's why I have transcribed (still in progression in 10/13/2020) by ear the 5 Deezer top songs of the following reggae artists : 

*Alborosie, Alpha Blondy, Bob Marley & The Wailers, Black Uhuru, Burning Spear, Chronixx, Danakil, Damian Marley, Dub Inc, Gentleman, Groundation, Gregory Isaacs, Ijahman Levi, Inner Circle, Katchafire, Ky-Mani Marley, Lee "Scratch" Perry, Matisyahu, Mellow Mood, Naâman, Peter Tosh, Raging Fyah, Protoje, Rebelution, Sizzla, SOJA, Steel Pulse, Stephen Marley, Stick Figure, Tiken Jah Fakoly, Tribal Seeds, UB40, Ziggy Marley*

The transcription contains the skank chord progression and the bassline.
