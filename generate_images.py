import os

for i in range(0, 20):
	x = 1.0 * i / 20
	print str(x)
	with open(".\\src\\main.rs.tmpt", "rt") as fin:
		with open(".\\src\\main.rs", "wt") as fout:
			for line in fin:
				fout.write(line.replace('xxx', str(x)))
	os.system("cargo run")
