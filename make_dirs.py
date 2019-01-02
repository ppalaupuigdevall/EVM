import os

train_root_dir = '/work/ppalau/Extreme_Value_Machine/feature_vectors_imagenet/alexnet/cosine/train'
test_root_dir = '/work/ppalau/Extreme_Value_Machine/feature_vectors_imagenet/alexnet/test/known_classes'

known_classes = \
['n03218198','n03670208','n02096294','n03388549','n01491361','n03110669','n01773157','n03998194','n02028035','n04553703','n02865351','n10565667',\
 'n02107683','n02099712','n03445777','n04311004','n04008634','n04550184','n04355338','n02488702','n04332243','n02860847','n02116738','n02086646',\
 'n02494079','n07613480','n03062245']

classes_to_remove = ['n02950826','n03075370','n04548280','n02804414','n07747607','n04596742','n02509815','n04118776','n03623198',\
 'n03379051','n02403003','n03134739','n03345487','n02102318','n03976657','n04476259','n01749939','n03063599','n02769748','n01580077','n03483316','n04590129','n04523525']

#for elemm in classes_to_remove:
#	os.rmdir(os.path.join(root_dir, elemm))

for elem in known_classes:
	os.mkdir(os.path.join(train_root_dir, elem))

# for elem in known_classes:
# 	llista = os.listdir(os.path.join(train_root_dir, elem))
# 	for i in range(0, int(0.4 * len(llista))):
# 		os.rename(os.path.join(train_root_dir, elem, llista[i]), os.path.join(test_root_dir, elem, llista[i]))