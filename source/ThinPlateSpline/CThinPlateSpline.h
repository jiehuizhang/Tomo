
#ifndef CTHINPLATESPLINE_H_
#define CTHINPLATESPLINE_H_
//#include <boost/python.hpp>
//#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
using namespace cv;


enum TPS_INTERPOLATION
{
        FORWARD_WARP,                                   /**< forward transformation (optional)  */
        BACK_WARP                                               /**< back warp transform (standard)             */
};

/**
**************************************************************************
@brief CThinPlateSpline

This is the implementation of the original thin plate splien algorithm
introduce by Bookstein in 1989. This class helps you to use the algorithm
easy and efficient. See the function comments for more information.
***************************************************************************/
class CThinPlateSpline {
public:
        /**
        **************************************************************************
        CThinPlateSpline standard constructor.

        The standard constructor will do nothing else than set all control flags
        to the initial state. These contral flags are only used internally.
        ***************************************************************************/
        CThinPlateSpline();


        /**
        **************************************************************************
        CThinPlateSpline constructor.

        One has the possibility to create an object and initializing it directly
        with corresponding point sets that will allow a direct use of the class.

        @param pS                       Reference to a STL vector that holds the points of
        the source image.

        @param pD                       Reference to a STL vector that holds the points of
        the target image.
        ***************************************************************************/
        CThinPlateSpline(const std::vector<Point>& pS, const std::vector<Point>& pD);


        /**
        **************************************************************************
        CThinPlateSpline standard destructor.

        The standard destructor does nothing. All used datastructures are self-
        maintaining.
        ***************************************************************************/

        ~CThinPlateSpline();

        /**
        **************************************************************************
        The interpolation function is used to derive the position of a point
        transformed by the pre-computed spline coefficients. One has the possibility
        to warp forward or backward. The interpolation function determines for itself
        if the maps are suitable for forward/backward warps and recomputes them if
        necessary

        The spline is evaluated by computing the minimization of the energy function
        defined in Bookstein's original paper. The minimization function for the
        two dimensional problem is
        /f$ f(x,y) = a_1 + a_x*x + a_y * y + \sum_{i=1}^n w_i U(|P_i-(x,y)|)/f$

        As one can easily see the computation of one transformed pixel takes /f$O(n^2)/f$
        operations where n is the number of corresponding points.

        @param p                        Reference to the point that is going to be
        transformed/interpolated

        @param tpsInter defines if the point p is interpolated forward or
        backward


        @return                                 The function returns the interpolation result as
        cv::Point-structure
        ***************************************************************************/
        Point interpolate(const Point& p, const TPS_INTERPOLATION tpsInter = BACK_WARP);

        /**
        **************************************************************************
        This function is used to add new corresponding points to the internal
        representation. When this function is called all internal states are reset
        so that the spline coefficients are recomputed.

        @param pS                       Reference to a point from the source image

        @param pD                       Reference to a point from the target image


        @return                                 void
        ***************************************************************************/
        void addCorrespondence(const Point& pS, const Point& pD);

        /**
        **************************************************************************
        This is a combination of all function of this class. The function is used
        to directly transform an image according to the corresponding points
        provided earlier. If not already done, the function will compute the
        spline coefficients and then compute the transformation maps which will
        be used to transform the source image into it's destination using the
        cv::remap function. It is possible to choose the interpolation type as
        well as the transformation direction.

        @param src                                      Reference to the source image. Nothing will be done
        with this array other than reading values.

        @param dst                                      Reference the transformation destination. If dst
        was allocated before, it will be released and
        initialized with the correct size, depth etc.

        @param interpolation            Reference the transformation destination. If dst
        was allocated before, it will be released and
        initialized with the correct size, depth etc.
        All known interpolation function from the
        OpenCV library are possible: <br>
        INTER_NEAREST, <br>
        INTER_LINEAR, <br>
        INTER_AREA, <br>
        INTER_CUBIC, <br>
        INTER_LANCZOS4

        @param tpsInter                         This parameter defines the direction of the
        interpolation function. Possible values are: <br>
        FORWARD_WARP, <br>
        BACK_WARP

        @return                                 void
        ***************************************************************************/
        void warpImage(const Mat& src, 
			Mat& dst, 
			float lambda, 
			const int interpolation ,
			const TPS_INTERPOLATION tpsInter);


        /**
        **************************************************************************
        This function is used to set the corresponding points directily. Instead
        of adding single point correspondences one can put hole vectors of points
        into this class, which are used to compute the spline coefficients.

        Make sure that both vectors have the same size. If this is not the case
        the computation will fail!

        @param pS                       Reference to a vector containing the points
        from the source image

        @param pD                       Reference to a vector containing the points
        from the source image

        @return                                 void
        ***************************************************************************/
        void setCorrespondences(const std::vector<Point>& pS, const std::vector<Point>& pD);
       
        /**
        **************************************************************************
        The getMaps function is used to get your hands on the x- and y-values
        that are used for the interpolation. The maps contain the sub-pixel values
        which are used by cv::remap to interpolate each pixel.

        @param mapx                             sub-pixel x-coordinates

        @param mapy                             sub-pixel y-coordinates

        @return                                 void
        ***************************************************************************/
        void getMaps(Mat& mapx, Mat& mapy);

        /**
        **************************************************************************
        If you want to get your hands on the maps to interpolate for yourself
        they have to be computed first. The following function will do this. The
        reason why getMaps() doesen't do this for itself is, that the getMaps
        function is supposed to only return the arrays without knowing if you want
        FORWARD- or BACK_WARP maps.

        @param dstSize                  size of the destination image. Here you can insert
        any size you like. The algorithm will simply
        evaluate the spline and look on the correct position
        in the array. If there is no value, the result will
        be black.

        @param mapx                             sub-pixel x-coordinates

        @param mapy                             sub-pixel y-coordinates

        @param tpsInter                 This parameter defines the direction of the
        interpolation function. Possible values are: <br>
        FORWARD_WARP, <br>
        BACK_WARP

        @return                                 void
        ***************************************************************************/
        void computeMaps(const Size& dstSize,
                Mat_<float>& mapx,
                Mat_<float>& mapy,
                const TPS_INTERPOLATION tpsInter = BACK_WARP);

private:

        /**
        ****************************************************************************
        \brief This is the main function of the thin plate spline algorithm.

        The function describes the surface on which a energy function will
        be minimized in the coefficient finding process. The function is defined
        as /f$ U(r) = -r^2 \log r^2/f$, where /f$ r = \sqrt(x^2+y^2)/f$ and is
        therefore the length of the vector connecting two corresponding points.

        @param  p1              First point (x,y) for the function computation
        @param  p2              First point (x,y) for the function computation

        @return                 the function will return a double value computed with the
        above mentioned function.

        ****************************************************************************/
        double fktU(const Point& p1, const Point& p2);

        void computeSplineCoeffs(std::vector<Point>& iP,
                std::vector<Point>& iiP,
                float lambda,
                const TPS_INTERPOLATION tpsInter = BACK_WARP);

        Point interpolate_back_(const Point& p);
        Point interpolate_forward_(const Point& p);


        Mat_<float> cMatrix;
        Mat_<float> mapx;
        Mat_<float> mapy;
        std::vector<Point> pSrc;
        std::vector<Point> pDst;

        // FLAGS
        bool FLAG_COEFFS_BACK_WARP_SET;
        bool FLAG_COEFFS_FORWARD_WARP_SET;
        bool FLAG_MAPS_FORWARD_SET;
        bool FLAG_MAPS_BACK_WARP_SET;

};

/* python wrapper using ctypes
#ifdef __cplusplus
extern "C"{
#endif

#ifdef TRIALDLL_EXPORT
#define TRIALDLL_API __declspec(dllexport)
#else
#define TRIALDLL_API __declspec(dllimport)
#endif

    TRIALDLL_API CThinPlateSpline* __stdcall new_CThinPlateSpline(const std::vector<Point>& pS, const std::vector<Point>& pD);

    TRIALDLL_API void __stdcall del_CThinPlateSpline(CThinPlateSpline *tpspline);

    TRIALDLL_API void __stdcall warpImage(CThinPlateSpline *tpspline, 
		const Mat& src, Mat& dst, float lambda, const int interpolation,const TPS_INTERPOLATION tpsInter);


#ifdef __cplusplus
}
#endif
*/

/* python wrapping using boost python
BOOST_PYTHON_MODULE(TPS)
{
	using namespace boost::python;

	class_<std::vector<Point>> >("PList")
        .def(vector_indexing_suite<std::vector<Point> >())
    ;

    class_<CThinPlateSpline>("CThinPlateSpline")
		.def(init<PList, PList>())
        .def("warpImage", &CThinPlateSpline::warpImage)
    ;
}
*/

#endif /* CTHINPLATESPLINE_H_ */
