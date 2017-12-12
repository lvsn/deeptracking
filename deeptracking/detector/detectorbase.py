import abc


class DetectorBase:
    @abc.abstractmethod
    def detect(self, img):
        """
        Return Transform from input image
        :param img:
        :return:
        """
        pass

    def get_likelihood(self):
        """
        return likelihood of last detection between 0 and 1 if
        :return:
        """
        return 1