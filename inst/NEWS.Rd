\name{NEWS}
\alias{NEWS}
\title{SimJoint updates}


\section{Changes in SimJoint version 0.3.4 (20200108)}{
  \itemize{
    \item Changed the random number generator from std::mt19937_64 to pcg64 <http://www.pcg-random.org/>.
  }
}


\section{Changes in SimJoint version 0.2.3 (20200102)}{
  \itemize{
    \item Fixed a reported bug. Forgot to recover the normalized distributions before \code{postSimOpt()} returns.
  }
}


\section{Changes in SimJoint version 0.2.2 (20190807)}{
  \itemize{
    \item Updated vignette.
  }
}


