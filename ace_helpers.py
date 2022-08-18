""" collection of various helper functions for running ACE"""

# from multiprocessing import dummy as multiprocessing
# import sys
# import os
# #sys.path.append('/lab/tmpig23/u/yao_code/Human-AI-Interface/')
# from matplotlib import pyplot as plt
# import matplotlib.gridspec as gridspec
# import tcav.model as model
# import numpy as np
# from PIL import Image
# from skimage.segmentation import mark_boundaries
# from sklearn import linear_model
# from torchvision import transforms
# from sklearn.model_selection import cross_val_score
# import tensorflow as tf
# import cv2
from model_wrapper import *
# import copy

def make_model(model_to_run, img_size,
               labels_path, device, gradcam_layer=None):
  """Make an instance of a model.

  Args:
    sess: tf session instance.
    model_to_run: a string that describes which model to make.
    model_path: Path to models saved graph.
    randomize: Start with random weights
    labels_path: Path to models line separated class names text file.

  Returns:
    a model instance.

  Raises:
    ValueError: If model name is not valid.
  """
  mymodel = PytorchModelWrapper_public(model_to_run, img_size, labels_path, device, gradcam_layer=gradcam_layer)
  return mymodel


def load_image_from_file(filename, shape):
  """Given a filename, try to open the file. If failed, return None.
  Args:
    filename: location of the image file
    shape: the shape of the image file to be scaled
  Returns:
    the image if succeeds, None if fails.
  Rasies:
    exception if the image was not the right shape.
  """
  if not tf.gfile.Exists(filename):
    tf.logging.error('Cannot find file: {}'.format(filename))
    return None
  try:
    img = np.array(Image.open(filename).resize(
        shape, Image.BILINEAR))
    #img = Image.open(filename)
    #preprocess = transforms.Compose([
    #  transforms.Resize(224),
    #  transforms.CenterCrop(224),
    #  transforms.ToTensor(),
      #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #    ])
    #img = np.array(preprocess(img))
    #print(img.shape) 
    #print(img)
    #img = np.transpose(img, (1, 2, 0))
    #Normalize pixel values to between 0 and 1.
    img = np.float32(img) / 255.0
    n_img = Norm(img)
    #print(img.shape)
    if not (len(img.shape) == 3 and img.shape[2] == 3):
      return None
    else:
      return n_img

  except Exception as e:
    tf.logging.info(e)
    return None
  return n_img

def Norm(img, mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
  n_img = copy.deepcopy(img)
  for i in range(3):
    n_img[:,:,i] = (img[:,:,i]-mean[i])/std[i]
  return n_img

def invNorm(img, mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
  n_img = copy.deepcopy(img)
  for i in range(3):
    n_img[:,:,i] = img[:,:,i]*std[i]+mean[i]
  return n_img

def load_images_from_files(filenames, max_imgs=500, return_filenames=False,
                           do_shuffle=True, run_parallel=True,
                           shape=(224, 224),
                           num_workers=100):
  """Return image arrays from filenames.
  Args:
    filenames: locations of image files.
    max_imgs: maximum number of images from filenames.
    return_filenames: return the succeeded filenames or not
    do_shuffle: before getting max_imgs files, shuffle the names or not
    run_parallel: get images in parallel or not
    shape: desired shape of the image
    num_workers: number of workers in parallelization.
  Returns:
    image arrays and succeeded filenames if return_filenames=True.
  """
  imgs = []
  # First shuffle a copy of the filenames.
  filenames = filenames[:]
  if do_shuffle:
    np.random.shuffle(filenames)
  if return_filenames:
    final_filenames = []
  if run_parallel:
    pool = multiprocessing.Pool(num_workers)
    imgs = pool.map(lambda filename: load_image_from_file(filename, shape),
                    filenames[:max_imgs])
    if return_filenames:
      final_filenames = [f for i, f in enumerate(filenames[:max_imgs])
                         if imgs[i] is not None]
    imgs = [img for img in imgs if img is not None]
  else:
    for filename in filenames:
      img = load_image_from_file(filename, shape)
      if img is not None:
        imgs.append(img)
        if return_filenames:
          final_filenames.append(filename)
      if len(imgs) >= max_imgs:
        break

  if return_filenames:
    return np.array(imgs), final_filenames
  else:
    return np.array(imgs)


def get_acts_from_images(imgs, model, bottleneck_name):
  """Run images in the model to get the activations.
  Args:
    imgs: a list of images
    model: a model instance
    bottleneck_name: bottleneck name to get the activation from
  Returns:
    numpy array of activations.
  """
  output = []
  for i in range(int(imgs.shape[0] / 32) + 1):
    if imgs[i * 32:(i + 1) * 32].shape[0] > 0:
      output.append(model.run_examples(imgs[i * 32:(i + 1) * 32], bottleneck_name))

  output = np.concatenate(output, 0)
  output = np.asarray(output).squeeze()
  return output


def flat_profile(cd, images, bottlenecks=None):
  """Returns concept profile of given images.

  Given a ConceptDiscovery class instance and a set of images, and desired
  bottleneck layers, calculates the profile of each image with all concepts and
  returns a profile vector

  Args:
    cd: The concept discovery class instance
    images: The images for which the concept profile is calculated
    bottlenecks: Bottleck layers where the profile is calculated. If None, cd
      bottlenecks will be used.

  Returns:
    The concepts profile of input images using discovered concepts in
    all bottleneck layers.

  Raises:
    ValueError: If bottlenecks is not in right format.
  """
  profiles = []
  if bottlenecks is None:
    bottlenecks = list(cd.dic.keys())
  if isinstance(bottlenecks, str):
    bottlenecks = [bottlenecks]
  elif not isinstance(bottlenecks, list) and not isinstance(bottlenecks, tuple):
    raise ValueError('Invalid bottlenecks parameter!')
  for bn in bottlenecks:
    profiles.append(cd.find_profile(str(bn), images).reshape((len(images), -1)))
  profile = np.concatenate(profiles, -1)
  return profile


def cross_val(a, b, methods):
  """Performs cross validation for a binary ResNet_pytorch task.

  Args:
    a: First class data points as rows
    b: Second class data points as rows
    methods: The sklearn ResNet_pytorch models to perform cross-validation on

  Returns:
    The best performing trained binary ResNet_pytorch odel
  """
  x, y = binary_dataset(a, b)
  best_acc = 0.
  if isinstance(methods, str):
    methods = [methods]
  best_acc = 0.
  for method in methods:
    temp_acc = 0.
    params = [10**e for e in [-4, -3, -2, -1, 0, 1, 2, 3]]
    for param in params:
      clf = give_classifier(method, param)
      acc = cross_val_score(clf, x, y, cv=min(100, max(2, int(len(y) / 10))))
      if np.mean(acc) > temp_acc:
        temp_acc = np.mean(acc)
        best_param = param
    if temp_acc > best_acc:
      best_acc = temp_acc
      final_clf = give_classifier(method, best_param)
  final_clf.fit(x, y)
  return final_clf, best_acc


def give_classifier(method, param):
  """Returns an sklearn ResNet_pytorch model.

  Args:
    method: Name of the sklearn ResNet_pytorch model
    param: Hyperparameters of the sklearn model

  Returns:
    An untrained sklearn ResNet_pytorch model

  Raises:
    ValueError: if the model name is invalid.
  """
  if method == 'logistic':
    return linear_model.LogisticRegression(C=param)
  elif method == 'sgd':
    return linear_model.SGDClassifier(alpha=param)
  else:
    raise ValueError('Invalid model!')


def binary_dataset(pos, neg, balanced=True):
  """Creates a binary dataset given instances of two classes.

  Args:
     pos: Data points of the first class as rows
     neg: Data points of the second class as rows
     balanced: If true, it creates a balanced binary dataset.

  Returns:
    The data points of the created data set as rows and the corresponding labels
  """
  if balanced:
    min_len = min(neg.shape[0], pos.shape[0])
    ridxs = np.random.permutation(np.arange(2 * min_len))
    x = np.concatenate([neg[:min_len], pos[:min_len]], 0)[ridxs]
    y = np.concatenate([np.zeros(min_len), np.ones(min_len)], 0)[ridxs]
  else:
    ridxs = np.random.permutation(np.arange(len(neg) + len(pos)))
    x = np.concatenate([neg, pos], 0)[ridxs]
    y = np.concatenate(
        [np.zeros(neg.shape[0]), np.ones(pos.shape[0])], 0)[ridxs]
  return x, y


def plot_concepts(cd, bn, num=10, address=None, mode='diverse', concepts=None):
  """Plots examples of discovered concepts.

  Args:
    cd: The concept discovery instance
    bn: Bottleneck layer name
    num: Number of images to print out of each concept
    address: If not None, saves the output to the address as a .PNG image
    mode: If 'diverse', it prints one example of each of the target class images
      is coming from. If 'radnom', randomly samples exmples of the concept. If
      'max', prints out the most activating examples of that concept.
    concepts: If None, prints out examples of all discovered concepts.
      Otherwise, it should be either a list of concepts to print out examples of
      or just one concept's name

  Raises:
    ValueError: If the mode is invalid.
  """
  if concepts is None:
    concepts = cd.dic[bn]['concepts']
  elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
    concepts = [concepts]
  num_concepts = len(concepts)
  plt.rcParams['figure.figsize'] = num * 2.1, 4.3 * num_concepts
  fig = plt.figure(figsize=(num * 2, 4 * num_concepts))
  outer = gridspec.GridSpec(num_concepts, 1, wspace=0., hspace=0.3)
  for n, concept in enumerate(concepts):
    if concept[-3:] == '999':
          continue 
    inner = gridspec.GridSpecFromSubplotSpec(
        2, num, subplot_spec=outer[n], wspace=0, hspace=0.1)
    concept_images = cd.dic[bn][concept]['images']
    concept_patches = cd.dic[bn][concept]['patches']
    concept_image_numbers = cd.dic[bn][concept]['image_numbers']
    if mode == 'max':
      idxs = np.arange(len(concept_images))
    elif mode == 'random':
      idxs = np.random.permutation(np.arange(len(concept_images)))
    elif mode == 'diverse':
      idxs = []
      while True:
        seen = set()
        for idx in range(len(concept_images)):
          if concept_image_numbers[idx] not in seen and idx not in idxs:
            seen.add(concept_image_numbers[idx])
            idxs.append(idx)
        if len(idxs) == len(concept_images):
          break
    else:
      raise ValueError('Invalid mode!')
    idxs = idxs[:num]
    for i, idx in enumerate(idxs):
      ax = plt.Subplot(fig, inner[i])
      ax.imshow(invNorm(concept_images[idx]))
      ax.set_xticks([])
      ax.set_yticks([])
      if i == int(num / 2):
        ax.set_title(concept)
      ax.grid(False)
      fig.add_subplot(ax)
      ax = plt.Subplot(fig, inner[i + num])
      mask = 1 - (np.mean(concept_patches[idx] == float(
          cd.average_image_value) / 255, -1) == 1)
      image = invNorm(cd.discovery_images[concept_image_numbers[idx]])
      #image = cd.discovery_images[concept_image_numbers[idx]]
      ax.imshow(mark_boundaries(image, mask, color=(1, 1, 0), mode='thick'))
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_title(str(concept_image_numbers[idx]))
      ax.grid(False)
      fig.add_subplot(ax)
  plt.suptitle(bn)
  if address is not None:
    with tf.io.gfile.GFile(address + bn + '.png', 'w') as f:
      fig.savefig(f)
    plt.clf()
    plt.close(fig)
def plot_match_concepts(cd, bn, num=10, address=None, mode='diverse', concepts=None):
  """Plots examples of discovered concepts.

  Args:
    cd: The concept discovery instance
    bn: Bottleneck layer name
    num: Number of images to print out of each concept
    address: If not None, saves the output to the address as a .PNG image
    mode: If 'diverse', it prints one example of each of the target class images
      is coming from. If 'radnom', randomly samples exmples of the concept. If
      'max', prints out the most activating examples of that concept.
    concepts: If None, prints out examples of all discovered concepts.
      Otherwise, it should be either a list of concepts to print out examples of
      or just one concept's name

  Raises:
    ValueError: If the mode is invalid.
  """
  if concepts is None:
    concepts = list(cd.center_match.keys())
  elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
    concepts = [concepts]
  num_concepts = len(concepts)
  plt.rcParams['figure.figsize'] = num * 2.1, 4.3 * num_concepts
  fig = plt.figure(figsize=(num * 2, 4 * num_concepts))
  outer = gridspec.GridSpec(num_concepts, 1, wspace=0., hspace=0.3)
  for n, concept in enumerate(concepts):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, num, subplot_spec=outer[n], wspace=0, hspace=0.1)
    index = cd.center_match[concept]
    concept_images = cd.dataset[index][np.newaxis,:] # (1, 224, 224, 3)
    concept_patches = cd.patches[index][np.newaxis,:]# (1, 224, 224, 3)
    concept_image_numbers = np.array([0]) #[1,]
    # concept_images = cd.dic[bn][concept]['images']
    # concept_patches = cd.dic[bn][concept]['patches']
    # concept_image_numbers = cd.dic[bn][concept]['image_numbers']
    if mode == 'max':
      idxs = np.arange(len(concept_images))
    elif mode == 'random':
      idxs = np.random.permutation(np.arange(len(concept_images)))
    elif mode == 'diverse':
      idxs = []
      while True:
        seen = set()
        for idx in range(len(concept_images)):
          if concept_image_numbers[idx] not in seen and idx not in idxs:
            seen.add(concept_image_numbers[idx])
            idxs.append(idx)
        if len(idxs) == len(concept_images):
          break
    else:
      raise ValueError('Invalid mode!')
    idxs = idxs[:num]
    for i, idx in enumerate(idxs):
      ax = plt.Subplot(fig, inner[i])
      ax.imshow(concept_images[idx])
      ax.set_xticks([])
      ax.set_yticks([])
      if i == int(num / 2):
        ax.set_title(concept)
      ax.grid(False)
      fig.add_subplot(ax)
      ax = plt.Subplot(fig, inner[i + num])
      mask = 1 - (np.mean(concept_patches[idx] == float(
          cd.average_image_value) / 255, -1) == 1)
      image = cd.discovery_images[concept_image_numbers[idx]]

      # im = Image.fromarray((image * 255).astype('uint8')).convert('RGB')
      # ma = Image.fromarray(mask.astype('uint8')).convert('RGBA')
      # im.paste(ma, box=None, mask=ma)




      ax.imshow(mark_boundaries(image, mask, color=(1, 1, 0), mode='thick'))
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_title(str(concept_image_numbers[idx]))
      ax.grid(False)
      fig.add_subplot(ax)
  plt.suptitle(bn)
  if address is not None:
    with tf.gfile.Open(address + bn + '.png', 'w') as f:
      fig.savefig(f)
    plt.clf()
    plt.close(fig)

def plot_target_match_concepts(cd, bn, target, tcav_score, num=1, address=None, mode='diverse', concepts=None):
  """Plots examples of discovered concepts.

  Args:
    cd: The concept discovery instance
    bn: Bottleneck layer name
    num: Number of images to print out of each concept
    address: If not None, saves the output to the address as a .PNG image
    mode: If 'diverse', it prints one example of each of the target class images
      is coming from. If 'radnom', randomly samples exmples of the concept. If
      'max', prints out the most activating examples of that concept.
    concepts: If None, prints out examples of all discovered concepts.
      Otherwise, it should be either a list of concepts to print out examples of
      or just one concept's name

  Raises:
    ValueError: If the mode is invalid.
  """
  if concepts is None:
    concepts = list(cd.center_match.keys())
  elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
    concepts = [concepts]
  '''choose target concept'''
  target_concept = []
  for tar in target:
    for n, concept in enumerate(concepts): # all concept
    # see if concept belongs to target concept
      if ('t'+ str(tar)+'_') in concept:
        target_concept.append(concept)
  # num_concepts = len(concepts)
  num_concepts = len(target)
  plt.rcParams['figure.figsize'] = (num_concepts + 1) * 4.3, 2.1 * num
  fig = plt.figure(figsize=( 4 * num_concepts, num * 2,))
  outer = gridspec.GridSpec(num, num_concepts + 1, wspace=0., hspace=0.3)
  '''color'''
  color_dict = [(0,1,1),  (0.5,1,0), (1,0,0), (1,1,1), (0,1,0)]
  image = cd.discovery_images[0]
  '''store the center point'''
  center_points = []
  for n, concept in enumerate(target_concept): # all concept
    '''only one row'''
    # inner = gridspec.GridSpecFromSubplotSpec(
    #     1, num_concepts + 1, subplot_spec=outer[n], wspace=0, hspace=0.1)  # only one row
    index = cd.center_match[concept]
    concept_images = cd.dataset[index] # (224, 224, 3)
    concept_patches = cd.patches[index]# (224, 224, 3)
    # concept_images = cd.dic[bn][concept]['images']
    # concept_patches = cd.dic[bn][concept]['patches']
    # concept_image_numbers = cd.dic[bn][concept]['image_numbers']

    ax = plt.Subplot(fig, outer[n+1])
    ax.imshow(concept_images)
    ax.set_xticks([])
    ax.set_yticks([])
    title = 'concept-{} tcav-{}'.format(n+1, tcav_score[n])
    ax.set_title(title, color=color_dict[n])
    ax.grid(False)
    fig.add_subplot(ax)
    mask = 1 - (np.mean(concept_patches == float(
        cd.average_image_value) / 255, -1) == 1)

    '''try to find the center of each concept'''
    contours, cnt = cv2.findContours(cv2.resize(mask.copy(), (224, 224)).astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])  # 计算第一条轮廓的各阶矩,字典形式
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    cv2.circle(image, (center_x, center_y), 10, color_dict[n], -1)  # 绘制中心点
    center_points.append((center_x, center_y))
    # cv2.imwrite("1.png", image)
    '''boundary drawing'''
    # image = mark_boundaries(image, mask, color=color_dict[n], mode='thick')
  # drawing the line
  for startpt in center_points:
    for endpt in center_points:
      if not startpt == endpt:
        cv2.line(image, startpt, endpt, (1, 1, 0), 5, 4)
  '''add the last one'''
  ax = plt.Subplot(fig, outer[0])

  ax.imshow(image)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_title('output')
  ax.grid(False)
  fig.add_subplot(ax)
  # plt.suptitle(bn)
  if address is not None:
    with tf.gfile.Open(address + bn + '.png', 'w') as f:
      fig.savefig(f)
    plt.clf()
    plt.close(fig)
def plot_target_match_concepts_batch(cd, bn, target, tcav_score, num=1, img_address=None, graph_address=None, filename=None, mode='diverse', concepts=None):
  """Plots examples of discovered concepts.

  Args:
    cd: The concept discovery instance
    bn: Bottleneck layer name
    num: Number of images to print out of each concept
    address: If not None, saves the output to the address as a .PNG image
    mode: If 'diverse', it prints one example of each of the target class images
      is coming from. If 'radnom', randomly samples exmples of the concept. If
      'max', prints out the most activating examples of that concept.
    concepts: If None, prints out examples of all discovered concepts.
      Otherwise, it should be either a list of concepts to print out examples of
      or just one concept's name

  Raises:
    ValueError: If the mode is invalid.
  """
  if concepts is None:
    concepts = list(cd.center_match.keys())
  elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
    concepts = [concepts]
  '''choose target concept'''
  target_concept = []
  for tar in target:
    for n, concept in enumerate(concepts): # all concept
    # see if concept belongs to target concept
      if ('t'+ str(tar)+'_') in concept:
        target_concept.append(concept)
  # num_concepts = len(concepts)
  num_concepts = len(target)
  plt.rcParams['figure.figsize'] = (num_concepts + 1) * 4.3, 2.1 * num
  fig = plt.figure(figsize=( 4 * num_concepts, num * 2,))
  outer = gridspec.GridSpec(num, num_concepts + 1, wspace=0., hspace=0.3)
  '''color'''
  color_dict = [(0,1,1),  (0.5,1,0), (1,0,0), (1,1,1), (0,1,0)]
  image = cd.discovery_images[0]
  '''store the center point'''
  center_points = []
  for n, concept in enumerate(target_concept): # all concept
    '''only one row'''
    # inner = gridspec.GridSpecFromSubplotSpec(
    #     1, num_concepts + 1, subplot_spec=outer[n], wspace=0, hspace=0.1)  # only one row
    index = cd.center_match[concept]
    concept_images = cd.dataset[index] # (224, 224, 3)
    concept_patches = cd.patches[index]# (224, 224, 3)
    # concept_images = cd.dic[bn][concept]['images']
    # concept_patches = cd.dic[bn][concept]['patches']
    # concept_image_numbers = cd.dic[bn][concept]['image_numbers']

    ax = plt.Subplot(fig, outer[n+1])
    ax.imshow(concept_images)
    ax.set_xticks([])
    ax.set_yticks([])
    title = 'concept-{} tcav-{}'.format(n+1, tcav_score[n])
    ax.set_title(title, color=color_dict[n])
    ax.grid(False)
    fig.add_subplot(ax)
    mask = 1 - (np.mean(concept_patches == float(
        cd.average_image_value) / 255, -1) == 1)

    '''try to find the center of each concept'''
    contours, cnt = cv2.findContours(cv2.resize(mask.copy(), (224, 224)).astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])  # 计算第一条轮廓的各阶矩,字典形式
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    cv2.circle(image, (center_x, center_y), 10, color_dict[n], -1)  # 绘制中心点
    center_points.append((center_x, center_y))
    # cv2.imwrite("1.png", image)
    '''boundary drawing'''
    # image = mark_boundaries(image, mask, color=color_dict[n], mode='thick')
  # drawing the line
  for startpt in center_points:
    for endpt in center_points:
      if not startpt == endpt:
        cv2.line(image, startpt, endpt, (1, 1, 0), 5, 4)
  '''add the last one'''
  ax = plt.Subplot(fig, outer[0])

  ax.imshow(image)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_title('output')
  ax.grid(False)
  fig.add_subplot(ax)
  # plt.suptitle(bn)

  '''save the final image'''
  if img_address is not None:
    with tf.gfile.Open(img_address +'/'+ filename +'_img'+ '.png', 'w') as f:
      fig.savefig(f)
    plt.clf()
    plt.close(fig)
  '''save the final graph node vectors (No distance) '''
  graph_dict = {}
  graph_distance_dict = {}
  for con in target_concept:
    # graph_dict[con] = cd.center_match_vector[con]
    graph_dict[con] = list(cd.center_match_vector[con])
  f = open(graph_address +'/'+ filename +'_graph'+ '.txt', 'w')
  f.write(str(graph_dict))
  f.close()

  '''save the final edge vectors'''
  edge_dict = {}
  for i, startpt in enumerate(center_points):
    edge_dict[i] = {}
    for j, endpt in enumerate(center_points):
      if not startpt == endpt:
        edge_dict[i][j] = tuple(np.array(list(startpt)) - np.array(list(endpt)))
  f = open(graph_address +'/'+ filename +'_edge'+ '.txt', 'w')
  f.write(str(edge_dict))
  f.close()

  # # # 读取
  # f = open(graph_address +'/'+ filename +'_edge'+ '.txt', 'r')
  # a = f.read()
  # dict_name = eval(a)
  # f.close()

def plot_target_match_concepts_test_dummy(cd, bn, target, tcav_score, num=1, img_address=None, graph_address=None, filename=None, mode='diverse', concepts=None):
  """Plots examples of discovered concepts.

  Args:
    cd: The concept discovery instance
    bn: Bottleneck layer name
    num: Number of images to print out of each concept
    address: If not None, saves the output to the address as a .PNG image
    mode: If 'diverse', it prints one example of each of the target class images
      is coming from. If 'radnom', randomly samples exmples of the concept. If
      'max', prints out the most activating examples of that concept.
    concepts: If None, prints out examples of all discovered concepts.
      Otherwise, it should be either a list of concepts to print out examples of
      or just one concept's name

  Raises:
    ValueError: If the mode is invalid.
  """
  if concepts is None:
    concepts = list(cd.center_match.keys())
  elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
    concepts = [concepts]
  '''choose target concept'''
  target_concept = []
  for tar in target:
    for n, concept in enumerate(concepts): # all concept
    # see if concept belongs to target concept
      if ('t'+ str(tar)+'_') in concept:
        target_concept.append(concept)
  # num_concepts = len(concepts)
  num_concepts = len(target)
  plt.rcParams['figure.figsize'] = (num_concepts + 1) * 4.3, 2.1 * num
  fig = plt.figure(figsize=( 4 * num_concepts, num * 2,))
  outer = gridspec.GridSpec(num, num_concepts + 1, wspace=0., hspace=0.3)
  '''color'''
  color_dict = [(0,1,1),  (0.5,1,0), (1,0,0), (0, 0, 0), (0,1,0)]
  image = cd.discovery_images[0]
  '''store the center point'''
  center_points = []
  for n, concept in enumerate(target_concept): # all concept
    '''only one row'''
    # inner = gridspec.GridSpecFromSubplotSpec(
    #     1, num_concepts + 1, subplot_spec=outer[n], wspace=0, hspace=0.1)  # only one row
    index = cd.center_match[concept]
    concept_images = cd.dataset[index] # (224, 224, 3)
    concept_patches = cd.patches[index]# (224, 224, 3)
    # concept_images = cd.dic[bn][concept]['images']
    # concept_patches = cd.dic[bn][concept]['patches']
    # concept_image_numbers = cd.dic[bn][concept]['image_numbers']

    ax = plt.Subplot(fig, outer[n+1])
    ax.imshow(concept_images)
    ax.set_xticks([])
    ax.set_yticks([])
    title = 'concept-{} tcav-{}'.format(n+1, tcav_score[n])
    ax.set_title(title, color=color_dict[n])
    ax.grid(False)
    fig.add_subplot(ax)
    mask = 1 - (np.mean(concept_patches == float(
        cd.average_image_value) / 255, -1) == 1)

    '''try to find the center of each concept'''
    contours, cnt = cv2.findContours(cv2.resize(mask.copy(), (224, 224)).astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])  # 计算第一条轮廓的各阶矩,字典形式
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    ''' see if you really detect the node'''
    if cd.center_match_error[concept] < 50000:  # this is a threshold
      cv2.circle(image, (center_x, center_y), 10, color_dict[n], -1)  # 绘制中心点
    else:
      cv2.circle(image, (center_x, center_y), 10, color_dict[4], -1)  # Use a special color
    center_points.append((center_x, center_y))
    # cv2.imwrite("1.png", image)
    '''boundary drawing'''
    # image = mark_boundaries(image, mask, color=color_dict[n], mode='thick')
  # drawing the line
  for startpt in center_points:
    for endpt in center_points:
      if not startpt == endpt:
        cv2.line(image, startpt, endpt, (1, 1, 0), 5, 4)
  '''add the last one'''
  ax = plt.Subplot(fig, outer[0])

  ax.imshow(image)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_title('output')
  ax.grid(False)
  fig.add_subplot(ax)
  # plt.suptitle(bn)

  '''save the final image'''
  if img_address is not None:
    with tf.gfile.Open(img_address +'/'+ filename +'_img'+ '.png', 'w') as f:
      fig.savefig(f)
    plt.clf()
    plt.close(fig)
  '''save the final graph node vectors (No distance) '''
  graph_dict = {}
  graph_distance_dict = {}
  for con in target_concept:
    # if cd.center_match_error[con] < 40000: # this is a threshold
    graph_dict[con] = list(cd.center_match_vector[con])
    # else:
    #   graph_dict[con] = [1.0] * 512
  f = open(graph_address +'/'+ filename +'_graph'+ '.txt', 'w')
  f.write(str(graph_dict))
  f.close()

  '''save the final edge vectors'''
  edge_dict = {}
  for i, startpt in enumerate(center_points):
    edge_dict[i] = {}
    for j, endpt in enumerate(center_points):
      if not startpt == endpt:
        edge_dict[i][j] = tuple(np.array(list(startpt)) - np.array(list(endpt)))
  f = open(graph_address +'/'+ filename +'_edge'+ '.txt', 'w')
  f.write(str(edge_dict))
  f.close()

  # # # 读取
  # f = open(graph_address +'/'+ filename +'_edge'+ '.txt', 'r')
  # a = f.read()
  # dict_name = eval(a)
  # f.close()
  # return graph_dict, edge_dict
def plot_target_match_concepts_test_dummy_edge(cd, num=1, img_address=None, graph_address=None, filename=None, mode='diverse', concepts=None):
  """Plots examples of discovered concepts.

  Args:
    cd: The concept discovery instance
    bn: Bottleneck layer name
    num: Number of images to print out of each concept
    address: If not None, saves the output to the address as a .PNG image
    mode: If 'diverse', it prints one example of each of the target class images
      is coming from. If 'radnom', randomly samples exmples of the concept. If
      'max', prints out the most activating examples of that concept.
    concepts: If None, prints out examples of all discovered concepts.
      Otherwise, it should be either a list of concepts to print out examples of
      or just one concept's name

  Raises:
    ValueError: If the mode is invalid.
  """
  if concepts is None:
    concepts = list(cd.center_match.keys())
  elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
    concepts = [concepts]
  '''choose target concept'''
  target_concept = concepts
  num_concepts = len(target_concept)

  '''build the synthesized image'''
  plt.rcParams['figure.figsize'] = (num_concepts + 1) * 4.3, 2.1 * num
  fig = plt.figure(figsize=( 4 * num_concepts, num * 2,))
  outer = gridspec.GridSpec(num, num_concepts + 1, wspace=0., hspace=0.3)
  '''color'''
  color_dict = [(0,1,1),  (0.5,1,0), (1,0,0), (1, 0, 1), (0, 0, 0) ]
  image = cd.discovery_images[0]


  '''store the center point'''
  center_points = []
  for n, concept in enumerate(target_concept): # all concept
    '''only one row'''
    # inner = gridspec.GridSpecFromSubplotSpec(
    #     1, num_concepts + 1, subplot_spec=outer[n], wspace=0, hspace=0.1)  # only one row
    index = cd.center_match[concept]
    concept_images = cd.dataset[index] # (224, 224, 3) Google (229, 229, 3) Xception
    concept_patches = cd.patches[index]# (224, 224, 3)
    # concept_images = cd.dic[bn][concept]['images']
    # concept_patches = cd.dic[bn][concept]['patches']
    # concept_image_numbers = cd.dic[bn][concept]['image_numbers']

    ax = plt.Subplot(fig, outer[n+1])
    ax.imshow(concept_images)
    ax.set_xticks([])
    ax.set_yticks([])
    title = 'concept-{}'.format(n+1)
    ax.set_title(title, color=color_dict[n])
    ax.grid(False)
    fig.add_subplot(ax)
    mask = 1 - (np.mean(concept_patches == float(
        cd.average_image_value) / 255, -1) == 1)

    '''try to find the center of each concept'''
    contours, cnt = cv2.findContours(cv2.resize(mask.copy(), (299, 299)).astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # (229, 229, 3) Xception
    M = cv2.moments(contours[0])  # 计算第一条轮廓的各阶矩,字典形式
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    ''' see if you really detect the node'''
    # if cd.center_match_error[concept] < 40000:  # this is a threshold # should be adjust due to the Xception
    cv2.circle(image, (center_x, center_y), 10, color_dict[n], -1)  # 绘制中心点
    # else:
    #   cv2.circle(image, (center_x, center_y), 10, color_dict[4], -1)  # Use a special color
    center_points.append((center_x, center_y))
    # cv2.imwrite("1.png", image)
    '''boundary drawing'''
    # image = mark_boundaries(image, mask, color=color_dict[n], mode='thick')
  # drawing the line
  for startpt in center_points:
    for endpt in center_points:
      if not startpt == endpt:
        cv2.line(image, startpt, endpt, (1, 1, 0), 5, 4)
  '''add the last one'''
  ax = plt.Subplot(fig, outer[0])
  ax.imshow(image)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_title('output')
  ax.grid(False)
  fig.add_subplot(ax)
  # plt.suptitle(bn)

  '''save the final image'''
  if img_address is not None:
    with tf.gfile.Open(img_address +'/'+ filename +'_img'+ '.png', 'w') as f:
      fig.savefig(f)
    plt.clf()
    plt.close(fig)
  '''save the final graph node vectors (No distance) '''
  graph_dict = {}
  graph_distance_dict = {}
  for con in target_concept:
    # if cd.center_match_error[con] < 40000: # this is a threshold
    graph_dict[con] = list(cd.center_match_vector[con])
    # else:
    #   graph_dict[con] = [1.0] * 2048 # for Xception
  f = open(graph_address +'/'+ filename +'_graph'+ '.txt', 'w')
  f.write(str(graph_dict))
  f.close()

  '''save the final edge vectors'''

  # edge_dict = []
  # for i, startpt in enumerate(center_points):
  #   edge_i = []
  #   for j, endpt in enumerate(center_points):
  #     if i == j: # not conect to it self
  #       edge_i.append(np.zeros((4)))
  #     else:
  #       '''normalized_center_points'''
  #       edge_i.append(np.concatenate((np.array(startpt), np.array(endpt))) / len(image) )
  #   edge_dict.append(edge_i)

  edge_dict = {}
  for i, startpt in enumerate(center_points):
    edge_dict[i] = {}
    for j, endpt in enumerate(center_points):
      if not startpt == endpt:
        edge_dict[i][j] = tuple(np.concatenate((np.array(startpt), np.array(endpt))) / len(image))
  f = open(graph_address +'/'+ filename +'_edge'+ '.txt', 'w')
  f.write(str(edge_dict))
  f.close()


  # # # 读取
  # f = open(graph_address +'/'+ filename +'_edge'+ '.txt', 'r')
  # a = f.read()
  # dict_name = eval(a)
  # f.close()
  # return graph_dict, edge_dict



def plot_target_match_topK_concepts(cd, bn, target, target_num_list,tcav_score, num=1, address=None, mode='diverse', concepts=None):
  """Plots examples of discovered concepts.

  Args:
    cd: The concept discovery instance
    bn: Bottleneck layer name
    num: Number of images to print out of each concept
    address: If not None, saves the output to the address as a .PNG image
    mode: If 'diverse', it prints one example of each of the target class images
      is coming from. If 'radnom', randomly samples exmples of the concept. If
      'max', prints out the most activating examples of that concept.
    concepts: If None, prints out examples of all discovered concepts.
      Otherwise, it should be either a list of concepts to print out examples of
      or just one concept's name

  Raises:
    ValueError: If the mode is invalid.
  """
  if concepts is None:
    concepts = list(cd.center_match.keys())
  elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
    concepts = [concepts]
  '''choose target concept'''
  target_concept = []
  for tar in target:
    for n, concept in enumerate(concepts): # all concept
    # see if concept belongs to target concept
      if ('t'+ str(tar)+'_') in concept:
        target_concept.append(concept)
  # num_concepts = len(concepts)
  num_concepts = len(target)
  plt.rcParams['figure.figsize'] = (sum(target_num_list) + 1) * 4.3, 2.1 * num
  fig = plt.figure(figsize=( 4 * sum(target_num_list), num * 2,))
  outer = gridspec.GridSpec(num, sum(target_num_list) + 1, wspace=0., hspace=0.3)
  '''color'''
  color_dict = [(0,1,1),  (0.5,1,0), (1,0,0), (1,1,1), (0,1,0)]
  image = cd.discovery_images[0]
  '''store the center point'''
  center_points = []
  subfig_num = 1
  for n, concept in enumerate(target_concept): # target 3 concept
    '''only one row'''
    # inner = gridspec.GridSpecFromSubplotSpec(
    #     1, num_concepts + 1, subplot_spec=outer[n], wspace=0, hspace=0.1)  # only one row
    instance_num = target_num_list[n]

    index = cd.center_match[concept][:instance_num]
    concept_images = cd.dataset[index][:instance_num] # (topk, 224, 224, 3)
    concept_patches = cd.patches[index][:instance_num]# (topk, 224, 224, 3)
    # concept_images = cd.dic[bn][concept]['images']
    # concept_patches = cd.dic[bn][concept]['patches']
    # concept_image_numbers = cd.dic[bn][concept]['image_numbers']

    for i in range(len(index)):
      '''maybe each concept have k instance like wheel'''
      ax = plt.Subplot(fig, outer[subfig_num])
      ax.imshow(concept_images[i])
      ax.set_xticks([])
      ax.set_yticks([])
      title = 'concept-{} tcav-{}'.format(n+1, tcav_score[n])
      ax.set_title(title, color=color_dict[n])
      ax.grid(False)
      fig.add_subplot(ax)
      subfig_num+=1
      mask = 1 - (np.mean(concept_patches[i] == float(
          cd.average_image_value) / 255, -1) == 1)

      '''try to find the center of each concept'''
      contours, cnt = cv2.findContours(cv2.resize(mask.copy(), (224, 224)).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      M = cv2.moments(contours[0])  # 计算第一条轮廓的各阶矩,字典形式
      center_x = int(M["m10"] / M["m00"])
      center_y = int(M["m01"] / M["m00"])
      cv2.circle(image, (center_x, center_y), 10, color_dict[n], -1)  # 绘制中心点
      center_points.append((center_x, center_y))
      # cv2.imwrite("1.png", image)
      '''boundary drawing'''
      # image = mark_boundaries(image, mask, color=color_dict[n], mode='thick')
  # drawing the line
  for startpt in center_points:
    for endpt in center_points:
      if not startpt == endpt:
        cv2.line(image, startpt, endpt, (1, 1, 0), 5, 4)
  '''add the last one'''
  ax = plt.Subplot(fig, outer[0])

  ax.imshow(image)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_title('output')
  ax.grid(False)
  fig.add_subplot(ax)
  # plt.suptitle(bn)
  if address is not None:
    with tf.gfile.Open(address + bn + '.png', 'w') as f:
      fig.savefig(f)
    plt.clf()
    plt.close(fig)

def cosine_similarity(a, b):
  """Cosine similarity of two vectors."""
  assert a.shape == b.shape, 'Two vectors must have the same dimensionality'
  a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)
  if a_norm * b_norm == 0:
    return 0.
  cos_sim = np.sum(a * b) / (a_norm * b_norm)
  return cos_sim


def similarity(cd, num_random_exp=None, num_workers=25):
  """Returns cosine similarity of all discovered concepts.

  Args:
    cd: The ConceptDiscovery module for discovered conceps.
    num_random_exp: If None, calculates average similarity using all the class's
      random concepts. If a number, uses that many random counterparts.
    num_workers: If greater than 0, runs the function in parallel.

  Returns:
    A similarity dict in the form of {(concept1, concept2):[list of cosine
    similarities]}
  """

  def concepts_similarity(cd, concepts, rnd, bn):
    """Calcualtes the cosine similarity of concept cavs.

    This function calculates the pairwise cosine similarity of all concept cavs
    versus an specific random concept

    Args:
      cd: The ConceptDiscovery instance
      concepts: List of concepts to calculate similarity for
      rnd: a random counterpart
      bn: bottleneck layer the concepts belong to

    Returns:
      A dictionary of cosine similarities in the form of
      {(concept1, concept2): [list of cosine similarities], ...}
    """
    similarity_dic = {}
    for c1 in concepts:
      cav1 = cd.load_cav_direction(c1, rnd, bn)
      for c2 in concepts:
        if (c1, c2) in similarity_dic.keys():
          continue
        cav2 = cd.load_cav_direction(c2, rnd, bn)
        similarity_dic[(c1, c2)] = cosine_similarity(cav1, cav2)
        similarity_dic[(c2, c1)] = similarity_dic[(c1, c2)]
    return similarity_dic

  similarity_dic = {bn: {} for bn in cd.bottlenecks}
  if num_random_exp is None:
    num_random_exp = cd.num_random_exp
  randoms = ['random_20class/random_20class{}'.format(i) for i in np.arange(num_random_exp)]
  concepts = {}
  for bn in cd.bottlenecks:
    concepts[bn] = [cd.target_class, cd.random_concept] + cd.dic[bn]['concepts']
  for bn in cd.bottlenecks:
    concept_pairs = [(c1, c2) for c1 in concepts[bn] for c2 in concepts[bn]]
    similarity_dic[bn] = {pair: [] for pair in concept_pairs}
    def t_func(rnd):
      return concepts_similarity(cd, concepts[bn], rnd, bn)
    if num_workers:
      pool = multiprocessing.Pool(num_workers)
      sims = pool.map(lambda rnd: t_func(rnd), randoms)
    else:
      sims = [t_func(rnd) for rnd in randoms]
    while sims:
      sim = sims.pop()
      for pair in concept_pairs:
        similarity_dic[bn][pair].append(sim[pair])
  return similarity_dic


def save_ace_report(cd, accs, scores, address):
  """Saves TCAV scores.

  Saves the average CAV accuracies and average TCAV scores of the concepts
  discovered in ConceptDiscovery instance.

  Args:
    cd: The ConceptDiscovery instance.
    accs: The cav accuracy dictionary returned by cavs method of the
      ConceptDiscovery instance
    scores: The tcav score dictionary returned by tcavs method of the
      ConceptDiscovery instance
    address: The address to save the text file in.
  """
  report = '\n\n\t\t\t ---CAV accuracies---'
  for bn in cd.bottlenecks:
    report += '\n'
    for concept in cd.dic[bn]['concepts']:
      report += '\n' + bn + ':' + concept + ':' + str(
          np.mean(accs[bn][concept]))
  with tf.gfile.Open(address, 'w') as f:
    f.write(report)
  report = '\n\n\t\t\t ---TCAV scores---'
  for bn in cd.bottlenecks:
    report += '\n'
    for concept in cd.dic[bn]['concepts']:
      pvalue = cd.do_statistical_testings(
          scores[bn][concept], scores[bn][cd.random_concept])
      report += '\n{}:{}:{},{}'.format(bn, concept,
                                       np.mean(scores[bn][concept]), pvalue)
  with tf.gfile.Open(address, 'w') as f:
    f.write(report)


def save_concepts(cd, concepts_dir):
  """Saves discovered concept's images or patches.

  Args:
    cd: The ConceptDiscovery instance the concepts of which we want to save
    concepts_dir: The directory to save the concept images
  """
  for bn in cd.bottlenecks:
    for concept in cd.dic[bn]['concepts']:
      if concept[-3:] == '999':
        continue
      patches_dir = os.path.join(concepts_dir, bn + '_' + concept + '_patches')
      images_dir = os.path.join(concepts_dir, bn + '_' + concept)
      patches = (np.clip(cd.dic[bn][concept]['patches'], 0, 1) * 256).astype(
          np.uint8)
      images = (np.clip(cd.dic[bn][concept]['images'], 0, 1) * 256).astype(
          np.uint8)
      tf.gfile.MakeDirs(patches_dir)
      tf.gfile.MakeDirs(images_dir)
      image_numbers = cd.dic[bn][concept]['image_numbers']
      image_addresses, patch_addresses = [], []
      for i in range(len(images)):
        image_name = '0' * int(np.ceil(2 - np.log10(i + 1))) + '{}_{}'.format(
            i + 1, image_numbers[i])
        patch_addresses.append(os.path.join(patches_dir, image_name + '.png'))
        image_addresses.append(os.path.join(images_dir, image_name + '.png'))
      save_images(patch_addresses, patches)
      save_images(image_addresses, images)


def save_images(addresses, images):
  """Save images in the addresses.

  Args:
    addresses: The list of addresses to save the images as or the address of the
      directory to save all images in. (list or str)
    images: The list of all images in numpy uint8 format.
  """
  if not isinstance(addresses, list):
    image_addresses = []
    for i, image in enumerate(images):
      image_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1) + '.png'
      image_addresses.append(os.path.join(addresses, image_name))
    addresses = image_addresses
  assert len(addresses) == len(images), 'Invalid number of addresses'
  for address, image in zip(addresses, images):
    with tf.gfile.Open(address, 'w') as f:
      Image.fromarray(invNorm(image)).save(f, format='PNG')


'''for each image, we process original and original- each concept'''

def get_concept_img(cd, target):
  '''

  target: [4, 1, 3, 5] index of concept
  '''
  concepts = list(cd.center_match.keys())
  target_concept = []
  for tar in target:
    for n, concept in enumerate(concepts): # all concept
    # see if concept belongs to target concept
    #   if ('t'+ str(tar)+'_') in concept:
      if (str(tar)) in concept:
        target_concept.append(concept)
    num_concepts = len(target)
  result = [cd.discovery_images.reshape([224,224,3])]
  for n, concept in enumerate(target_concept):
    index = cd.center_match[concept]
    concept_patches = cd.patches[index]
    concept_mask = 1 - cd.mask[index]
    concept_mask = concept_mask.reshape([224,224,1])
    img_original = cd.discovery_images.reshape([224,224,3])
    concept_mask = np.concatenate([concept_mask,concept_mask,concept_mask],axis = 2)
    img_delete_concept = img_original * concept_mask
    result.append(img_delete_concept)
  result = np.array(result)
  return result

'''detect activation from'''
def get_linears_from_images(img, model):
  '''
  img N * 224 * 224 * 3
  model cd.model

  '''
  # return np.asarray(get_linears(img, model)).squeeze()
  return np.asarray(get_linears(img, model))

def get_linears(examples, model):
  return model.get_linears(examples)
# def get_linears(examples, model):
#   return model.run_examples(examples, 'predictions')
def cosine_similarity(a, b):
  """Cosine similarity of two vectors."""
  assert a.shape == b.shape, 'Two vectors must have the same dimensionality'
  a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)
  if a_norm * b_norm == 0:
    return 0.
  cos_sim = np.sum(a * b) / (a_norm * b_norm)
  return cos_sim
