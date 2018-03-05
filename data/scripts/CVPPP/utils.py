import lmdb
import numpy as np


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


def create_dataset(
        output_path,
        image_paths,
        semantic_annotation_paths,
        instance_annotation_paths):

    n_images = len(image_paths)

    assert(n_images == len(semantic_annotation_paths))

    print 'Number of Images : {}'.format(n_images)

    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    n_images_cntr = 1
    for i in xrange(n_images):
        image_path = image_paths[i]
        semantic_annotation_path = semantic_annotation_paths[i]
        instance_annotation_path = instance_annotation_paths[i]

        image = open(image_path, 'r').read()

        semantic_annotation = np.load(semantic_annotation_path)
        semantic_annotation_height = semantic_annotation.shape[0]
        semantic_annotation_width = semantic_annotation.shape[1]

        instance_annotation = np.load(instance_annotation_path)

        n_objects = instance_annotation.shape[2]

        cache['image-{}'.format(n_images_cntr)] = image
        cache['semantic-annotation-{}'.format(n_images_cntr)
              ] = semantic_annotation.tostring()
        cache['instance-annotation-{}'.format(n_images_cntr)
              ] = instance_annotation.tostring()
        cache['height-{}'.format(n_images_cntr)
              ] = str(semantic_annotation_height)
        cache['width-{}'.format(n_images_cntr)
              ] = str(semantic_annotation_width)
        cache['n_objects-{}'.format(n_images_cntr)] = str(n_objects)

        if n_images_cntr % 50 == 0:
            write_cache(env, cache)
            cache = {}
            print 'Processed %d / %d' % (n_images_cntr, n_images)
        n_images_cntr += 1

    cache['num-samples'] = str(n_images)
    write_cache(env, cache)
    print 'Created dataset with {} samples'.format(n_images)
