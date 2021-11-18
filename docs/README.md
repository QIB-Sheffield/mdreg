**Updating the Reference Manual:** 

First, run the command `pdoc --html --force --output-dir "docs" "MDR"` in the parent folder.

Then, move all files in "docs/MDR" to "docs" folder and delete the "MDR" folder. Alternatively, you can type the following in the terminal:

`mv docs/MDR/* docs`

`rmdir docs/MDR`

Finally, commit the newly generated files to Github so that the [website](https://qib-sheffield.github.io/MDR-Library/) is refreshed.
