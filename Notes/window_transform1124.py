def window_transform(ct_array, windowWidth, windowCenter, normal=False):
	"""
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
	minWindow = float(windowCenter) - 0.5*float(windowWidth)
	newimg = (ct_array - minWindow) / float(windowWidth)
	newimg[newimg < 0] = 0
	newimg[newimg > 1] = 1
	if not normal:
		newimg = (newimg * 255).astype('uint8')
	return newimg
