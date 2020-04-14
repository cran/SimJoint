\name{NEWS}
\alias{NEWS}
\title{SimJoint updates}


\section{Changes in SimJoint version 0.3.6 (20200331)}{
  \itemize{
    \item Upon the request from Brian D. Ripley, time-consuming examples are commented rather than wrapped by \\donttest\{\}.
  }
}


\section{Changes in SimJoint version 0.3.5 (20200112)}{
  \itemize{
    \item \code{r-patched-solaris-x86} generated compile error due to the fact that the header file \code{/usr/include/floatingpoint.h} used variable name \code{extended}, which is re-declared in \code{SimJoint/src/pcg/pcg_random.hpp}. Changed \code{extended} to \code{pcgExtended} in the latter file.
  }
}


\section{Changes in SimJoint version 0.3.4 (20200108)}{
  \itemize{
    \item Changed the random number generator from \code{std::mt19937_64} to \code{pcg64} <http://www.pcg-random.org/>.
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


