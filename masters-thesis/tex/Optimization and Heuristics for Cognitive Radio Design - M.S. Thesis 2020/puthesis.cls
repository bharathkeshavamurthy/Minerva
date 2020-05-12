%
%  puthesis.cls  2017-04-15  Mark Senn  http://engineering.purdue.edu/~mark
%
%  INDEX: Purdue University thesis document class
%
%  DESCRIPTION:
%
%    This is a LaTeX document class for Purdue University theses.
%
%  USAGE:
%
%    See http://engineering.purdue.edu/~mark/puthesis for more information.
%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{puthesis}[2017/03/06 Purdue thesis class, version 1.9.23]
\usepackage{ifthen}
\newcommand{\ifthen}[2]{\ifthenelse{#1}{#2}{}}
\usepackage{endnotes}
\usepackage{pulongtable}
\usepackage{rotating}
\usepackage{lscape}
\usepackage[letterpaper]{geometry}

\makeatletter

\def\@@number{\string##}

\newcount{\@@i}
\newcounter{@@tcount}
\newcounter{@@volume}
\newcounter{last@@volume}
\newcounter{save@@page}
\newlength{\@@captionwidth}  
\newlength{\@@parindent}  \setlength{\@@parindent}{\parindent}
\newlength{\@@padding}
\newlength{\@@tlength}
\newlength{\@@ulength}
\newcommand{\@@dept}{unknown}
\newcommand{\set@@dept}[1]{\renewcommand{\@@dept}{#1}}

\newcommand{\@@Repeat}[2]{%
    \@@i=0
    \loop
        \ifnum\@@i<#2
        #1
        \advance\@@i by 1
    \repeat
}

\newcommand{\@@blankpage}{%
  \clearpage
  \mbox{}\clearpage
}

\newcommand{\articlepages}[1]{%
    \@@Repeat{\@@blankpage}{#1}
}

\newif\if@@more
\@@moretrue
\newcommand{\articleinclude}[1]{%
    \def\@@t{#1}
    \@@i=0
    \loop
        \advance\@@i by 1
        \def\@@u{\@@t\the\@@i.eps}
        \immediate\write16{\@@u}
        \IfFileExists{\@@u}{\noindent\includegraphics[width=\textwidth]{\@@u}\newpage}{\@@morefalse}
        \if@@more\repeat
}

\DeclareOption{iupuiece}{\set@@dept{iupuiece}}
\DeclareOption{aae}{\set@@dept{aae}}
\DeclareOption{agecon}{\set@@dept{agecon}}
\DeclareOption{agry}{\set@@dept{agry}}
\DeclareOption{ansc}{\set@@dept{ansc}}
\DeclareOption{che}{\set@@dept{che}}
\DeclareOption{ce}{\set@@dept{ce}}
\DeclareOption{edci}{\set@@dept{edci}}
\DeclareOption{ece}{\set@@dept{ece}}
\DeclareOption{hsci}{\set@@dept{hsci}}
\DeclareOption{ie}{\set@@dept{ie}}
\DeclareOption{ling}{\set@@dept{ling}}
\DeclareOption{mgmt}{\set@@dept{mgmt}}
\DeclareOption{mse}{\set@@dept{mse}}
\DeclareOption{me}{\set@@dept{me}}
\DeclareOption{ne}{\set@@dept{ne}}
\DeclareOption{chem}{\set@@dept{chem}}
\DeclareOption{cs}{\set@@dept{cs}}
\DeclareOption{eas}{\set@@dept{eas}}
\DeclareOption{math}{\set@@dept{math}}
\DeclareOption{phys}{\set@@dept{phys}}
\DeclareOption{stat}{\set@@dept{stat}}
\newcommand{\pendnotes}{}
\DeclareOption{endnotes}{%
  \let\footnote=\endnote
  \renewcommand{\pendnotes}{%
    \newpage
    \begingroup
%% !  \renewcommand{\baselinestretch}{0.9}
      \setlength{\parindent}{0pt}
      \setlength{\parskip}{1.5ex}
      \renewcommand{\enotesize}{\normalsize}
      \theendnotes
    \endgroup
  }
}
\newcommand{\@@type}{unknown}
\newcommand{\set@@type}[1]{\renewcommand{\@@type}{#1}}
\DeclareOption{bypass}{\set@@type{bypass}}
\DeclareOption{dissertation}{\set@@type{dissertation}}
\DeclareOption{prelim}{\set@@type{prelim}}
\DeclareOption{thesis}{\set@@type{thesis}}
\newboolean{@@uglyheadings}
\setboolean{@@uglyheadings}{false}
\DeclareOption{uglyheadings}{\setboolean{@@uglyheadings}{true}}
\newboolean{@@unset}
\newcommand{\@@optionbibstyle}{}
\newcommand{\set@@optionbibstyle}[1]{\renewcommand{\@@optionbibstyle}{#1}\addtocounter{@@tcount}{1}}
\DeclareOption{abbrv}{\set@@optionbibstyle{abbrv}}
\DeclareOption{abbrvnat}{\set@@optionbibstyle{abbrvnat}}
\DeclareOption{aer}{\set@@optionbibstyle{aer}}
\DeclareOption{agsm}{\set@@optionbibstyle{agsm}}
\DeclareOption{aip}{\set@@optionbibstyle{aip}}
\DeclareOption{alpha}{\set@@optionbibstyle{alpha}}
\DeclareOption{ama}{\set@@optionbibstyle{ama}}
\DeclareOption{apacite}{\set@@optionbibstyle{apacite}}
\DeclareOption{apalike}{\set@@optionbibstyle{apalike}}
\DeclareOption{astron}{\set@@optionbibstyle{astron}}
\DeclareOption{chicago}{\set@@optionbibstyle{chicago}}
\DeclareOption{ieee}{\set@@optionbibstyle{ieee}}
\DeclareOption{IEEEtran}{\set@@optionbibstyle{IEEEtran}}
\DeclareOption{jfm}{\set@@optionbibstyle{jfm}}
\DeclareOption{jfm2}{\set@@optionbibstyle{jfm2}}
\DeclareOption{kluwer}{\set@@optionbibstyle{kluwer}}
\DeclareOption{phaip}{\set@@optionbibstyle{phaip}}
\DeclareOption{plain}{\set@@optionbibstyle{plain}}
\DeclareOption{plainnat}{\set@@optionbibstyle{plainnat}}
\DeclareOption{unsrt}{\set@@optionbibstyle{unsrt}}
\DeclareOption{unsrtnat}{\set@@optionbibstyle{unsrtnat}}
\newif\if@openbib
\@openbibfalse
%
\newboolean{@@nonchapterblankpages}
\setboolean{@@nonchapterblankpages}{false}
\newcommand{\nonchapterblankpages}{\setboolean{@@nonchapterblankpages}{true}}
\newcommand{\nononchapterblankpages}{\setboolean{@@nonchapterblankpages}{false}}
\DeclareOption{nonchapterblankpages}{\nonchapterblankpages}
\DeclareOption{nononchapterblankpages}{\nononchapterblankpages}
%
\newboolean{@@chapterblankpages}
\setboolean{@@chapterblankpages}{true}
\newcommand{\chapterblankpages}{\setboolean{@@chapterblankpages}{true}}
\newcommand{\nochapterblankpages}{\setboolean{@@chapterblankpages}{false}}
\DeclareOption{chapterblankpages}{\chapterblankpages}
\DeclareOption{nochapterblankpages}{\nochapterblankpages}
%
\newboolean{@@coversheets}
\setboolean{@@coversheets}{true}
\newcommand{\coversheets}{\setboolean{@@coversheets}{true}}
\newcommand{\nocoversheets}{\setboolean{@@coversheets}{false}}
\DeclareOption{coversheets}{\coversheets}
\DeclareOption{nocoversheets}{\nocoversheets}
%
\DeclareOption{miser}{
  \nononchapterblankpages
  \nochapterblankpages
  \nocoversheets
}
\DeclareOption{nomiser}{
  \nonchapterblankpages
  \chapterblankpages
  \coversheets
}

\newboolean{number@@all@@volumes}
\setboolean{number@@all@@volumes}{false}
\DeclareOption{numberallvolumes}{\setboolean{number@@all@@volumes}{true}}
\DeclareOption{nonumberallvolumes}{\setboolean{number@@all@@volumes}{false}}

%\DeclareOptions*(\PassOptionsToClass{\CurrentOption}{report}}
% Used to count number of citation/bibliography styles used.
\setcounter{@@tcount}{0}
\ProcessOptions
\newcommand{\IW}[1]{\immediate\write16{#1}}
{
  \catcode`\+=13
  \def+{\space}
  \ifthen{\equal{\@@dept}{unknown}}
    {
      \IW{}
      \IW{You must specify an option for your school or department, for example,}
      \IW{++++\string\documentclass[aae]{puthesis}}
      \IW{The available school and department options are defined at}
      \IW{++++http://engineering.purdue.edu/\string~mark/puthesis/\@@number Options}
      \IW{ABORTING...}
      \IW{}
      \stop
    }
  \ifthen{\equal{\@@type}{unknown}}
    {
      \IW{}
      \IW{You must specify an option for your type of document, for example,}
      \IW{++++\string\documentclass[aae,dissertation]{puthesis}}
      \IW{The available document types are}
      \IW{++++bypass+++++++++Master's Bypass Report}
      \IW{++++dissertation+++Ph.D. dissertation}
      \IW{++++prelim+++++++++Ph.D. Preliminary Report}
      \IW{++++thesis+++++++++Master's Thesis}
      \IW{ABORTING...}
      \IW{}
      \stop
    }
  \ifnum\value{@@tcount}>1
    \IW{}
    \IW{You may specify only one citation/bibliography style from the below list:}
    \IW{++++abbrv}
    \IW{++++abbrvnat}
    \IW{++++agsm}
    \IW{++++aip}
    \IW{++++alpha}
    \IW{++++ama}
    \IW{++++apacite}
    \IW{++++apalike}
    \IW{++++astron}
    \IW{++++chicago}
    \IW{++++ieee}
    \IW{++++IEEEtran}
    \IW{++++jfm}
    \IW{++++jfm2}
    \IW{++++kluwer}
    \IW{++++phaip}
    \IW{++++plain}
    \IW{++++plainnat}
    \IW{++++unsrt}
    \IW{++++unsrtnat}
    \IW{See}
    \IW{++++http://engineering.purdue.edu/\string~mark/puthesis/\@@number Options}
    \IW{for more details.}
    \IW{ABORTING...}
    \IW{}
    \stop
  \fi
}
\LoadClass[12pt,twoside]{report}[2004/02/16]

% Got these lines using "grep dotted report.cls" and changed
% as needed for new \@dottedtocline \vskip parameter.
\renewcommand*\l@section{\@dottedtocline{1}{\smalltocskip}{1.5em}{2.3em}}
\renewcommand*\l@subsection{\@dottedtocline{2}{\smalltocskip}{3.8em}{3.2em}}
\renewcommand*\l@subsubsection{\@dottedtocline{3}{\smalltocskip}{7.0em}{4.1em}}
\renewcommand*\l@paragraph{\@dottedtocline{4}{\smalltocskip}{10em}{5em}}
\renewcommand*\l@subparagraph{\@dottedtocline{5}{\smalltocskip}{12em}{6em}}
\renewcommand*\l@figure{\@dottedtocline{1}{\bigtocskip}{0em}{2.3em}}
\renewcommand*\l@table{\@dottedtocline{1}{\bigtocskip}{0em}{2.3em}}

%{\obeyspaces
%  \gdef\@@Identification{{
%    \IW{}
%    \IW{                                      ||}
%    \IW{                                  see || below}
%    \IW{                                      ||}
%    \IW{                                   \string\\ || //}
%    \IW{                                    \string\\||//}
%    \IW{                                     \string\\//}
%    \IW{+----------------------------------------------------------------------------+}
%    \IW{| This document is using the 2017/03/06 version of puthesis.cls.  It was     |}
%    \IW{| probably read in because you used \string\documentclass[...]{puthesis}.           |}
%    \IW{|                                                                            |}
%    \IW{| Only the latest version of this file available from                        |}
%    \IW{|     http://engineering.purdue.edu/\string~mark/puthesis                               |}
%    \IW{| is supported.  Put it in the same subdirectory as your thesis.             |}
%    \IW{|                                                                            |}
%    \IW{| This file may be being read from your thesis directory or somewhere else.  |}
%    \IW{| To find out which is being used search for ``puthesis.cls'' near the       |}
%    \IW{| beginning of this log.  If it has ``(puthesis.cls'' or ``(./puthesis.cls'' |}
%    \IW{| it's from your thesis directory, otherwise it is being read form somewhere |}
%    \IW{| else.                                                                      |}
%    \IW{+----------------------------------------------------------------------------+}
%    \IW{                                     //\string\\}
%    \IW{                                    //||\string\\}
%    \IW{                                   // || \string\\}
%    \IW{                                      ||}
%    \IW{                                  see || above}
%    \IW{                                      ||}
%    \IW{}
%}}
%}

\def\AppendixFigure{\relax}
\def\AppendixTable{\relax}
\def\@@bibname{REFERENCES}
\def\@@deptbibstyle{unsrt}
\def\@@coversheetspace{\vfill}
\def\@@evenfoot{}
\def\@@evenhead{\hfil\textrm{\thepage}}
\newboolean{@@figurecaptions}
  \setboolean{@@figurecaptions}{false}
\def\@@oddfoot{}
\def\@@oddhead{\hfil\textrm{\thepage}}
\def\@@sectionbaselinestretch{\relax}
\def\@@sectionseries{\bfseries}
\def\@@startthebibliography{\coversheet{\@@bibname}}
\def\@@startvita{\coversheet{\@@vitaname}}
\def\@@subsectionbaselinestretch{\relax}
\def\@@subsectionseries{\bfseries}
\def\@@subsubsectionbaselinestretch{\relax}
\def\@@subsubsectionseries{\bfseries}
\def\@@t{\relax}
\def\@@thebibliographyparsep{\relax}

\ifthen{\equal{iupuiece}{\@@dept}}
  {
    \def\@@makechapterhead#1{
      \ifthenelse{\boolean{@@inappendix}}
        {\large\bf APPENDIX \thechapter\\\uppercase{#1}}
        {\large\bf\thechapter. \uppercase{#1}}
    }
    \def\@@makechapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
    \def\@@startvita{\relax}
  }
\ifthen{\equal{aae}{\@@dept}}
  {
    \def\fnum@table{\tablename~\thetable.~}
    \setboolean{@@figurecaptions}{true}
    \def\@@makechapterhead#1{\large\bf\thechapter. #1}
    \def\@@makechapterheadspacea{\vspace*{0.5truein}}
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.5truein}}
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
  }
\ifthen{\equal{agecon}{\@@dept}}
  {
    \def\@@makechapterhead#1{\rm\uppercase{Chapter \thechapter. #1}}
    \def\@@makechapterheadspacea{\vspace*{0.6truein}}
    \def\@@makechapterheadspaceb{}
    \def\@@makeschapterheadspacea{\vspace*{0.6truein}}
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
  }
%
%  The agry option is being built to be most like the plain Purdue standards.
%  It was originally based on the CS format.
\ifthen{\equal{agry}{\@@dept}}
  {
    \def\@@makechapterhead#1{\thechapter\quad\uppercase{#1}}
    \def\@@makechapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
    \def\@@sectionseries{}
    \def\@@subsectionseries{}
    \def\@@subsubsectionseries{}
  }
\ifthen{\equal{ansc}{\@@dept}}
  {
    \def\@@makechapterhead#1{
      \ifthenelse{\boolean{@@inappendix}}
        {\large\bf APPENDIX \thechapter\\\uppercase{#1}}
        {\large\bf\thechapter. \uppercase{#1}}
    }
    \def\@@makechapterheadspacea{\vspace*{0.5truein}}
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.5truein}}
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
    \def\@@startvita{\relax}
  }
\ifthen{\equal{che}{\@@dept}}
  {
    \def\@@makechapterhead#1{\thechapter. \uppercase{#1}}
    \def\@@makechapterheadspacea{\vspace*{0.65625truein}}
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.65625truein}}
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
    \def\@@sectionbaselinestretch{\renewcommand{\baselinestretch}{1}}
    \def\@@sectionseries{}
    \def\@@subsectionbaselinestretch{\renewcommand{\baselinestretch}{1}}
    \def\@@subsectionseries{\relax}
    \def\@@subsubsectionbaselinestretch{\renewcommand{\baselinestretch}{1}}
    \def\@@subsubsectionseries{\relax}
  }
\ifthen{\equal{ce}{\@@dept}}
  {
    \def\fnum@table{\tablename~\thetable.~}
    \setboolean{@@figurecaptions}{true}
    \def\@@makechapterhead#1{\large\rm\thechapter. \uppercase{#1}}
    \def\@@makechapterheadspacea{\vspace*{0.5truein}}
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.5truein}}
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
    \def\@@subsectionseries{\normalfont}
    \def\@@subsubsectionseries{\normalfont}
  }
\ifthen{\equal{edci}{\@@dept}}
  {
    \def\@@deptbibstyle{apalike}
    \def\@@makechapterhead#1{\large\bf\thechapter. #1}
    \def\@@makechapterheadspacea{\vspace*{0.5truein}}
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.5truein}}
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
    \def\@@startthebibliography{}
    \def\@@startvita{\relax}
  }
\ifthen{\equal{ece}{\@@dept}}
  {
    \def\@@deptbibstyle{IEEEtran}
    \renewcommand*\l@section{\@dottedtocline{1}{\bigtocskip}{1.5em}{2.3em}}
    \renewcommand*\l@subsection{\@dottedtocline{2}{\bigtocskip}{3.8em}{3.2em}}
    \renewcommand*\l@subsubsection{\@dottedtocline{3}{\bigtocskip}{7.0em}{4.1em}}
    \renewcommand*\l@paragraph{\@dottedtocline{4}{\bigtocskip}{10em}{5em}}
    \renewcommand*\l@subparagraph{\@dottedtocline{5}{\bigtocskip}{12em}{6em}}
    \renewcommand*\l@figure{\@dottedtocline{1}{\bigtocskip}{1.5em}{2.3em}}
    \def\@@makechapterhead#1{{\large\bf\thechapter. \uppercase{#1}}}
    \def\@@makechapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterhead#1{{\large\bf\uppercase{#1}}}
    \def\@@makeschapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
    \nochapterblankpages
  }
\ifthen{\equal{hsci}{\@@dept}}
  {
    \def\@@makechapterhead#1{
      \ifthenelse{\boolean{@@inappendix}}
        {\large\bf APPENDIX \thechapter\\\uppercase{#1}}
        {\large\bf\thechapter. \uppercase{#1}}
    }
    \def\@@makechapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
    \nochapterblankpages
  }
\ifthen{\equal{ie}{\@@dept}}
  {
    \def\@@makechapterhead#1{
      \ifthenelse{\boolean{@@inappendix}}
        {\large\bf APPENDIX \thechapter\\\uppercase{#1}}
        {\large\bf\thechapter. \uppercase{#1}}
    }
    \def\@@makechapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
    \nochapterblankpages
  }
\ifthen{\equal{ling}{\@@dept}}
{
    \def\@@deptbibstyle{unified}
    \renewcommand*\l@section{\@dottedtocline{1}{\bigtocskip}{1.5em}{2.3em}}
    \renewcommand*\l@subsection{\@dottedtocline{2}{\bigtocskip}{3.8em}{3.2em}}
    \renewcommand*\l@subsubsection{\@dottedtocline{3}{\bigtocskip}{7.0em}{4.1em}}
    \renewcommand*\l@paragraph{\@dottedtocline{4}{\bigtocskip}{10em}{5em}}
    \renewcommand*\l@subparagraph{\@dottedtocline{5}{\bigtocskip}{12em}{6em}}
    \renewcommand*\l@figure{\@dottedtocline{1}{\bigtocskip}{1.5em}{2.3em}}
    \def\@@makechapterhead#1{{\thechapter. \uppercase{#1}}}
    \def\@@makechapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterhead#1{{\uppercase{#1}}}
    \def\@@makeschapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
    \nochapterblankpages
}
\ifthen{\equal{mgmt}{\@@dept}}
  {
    \def\@@makechapterhead#1{\large\bf\thechapter. #1}
    \def\@@makechapterheadspacea{\vspace*{0.5truein}}
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.5truein}}
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
  }
\ifthen{\equal{mse}{\@@dept}}
  {
    \def\AppendixFigure{\addtocontents{lof}{Appendix Figure\hfil}}
    \def\AppendixTable{\addtocontents{lot}{Appendix Table\hfil}}
    \def\@@coversheetspace{\vspace*{0.53125truein}}
    \def\@@makechapterhead#1{
      \rm
      \ifthen{\boolean{@@inappendix}}
        {APPENDIX }
      \thechapter. \uppercase{#1}
    }
    \def\@@makechapterheadspacea{\vspace*{0.65625truein}}
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.65625truein}}
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
    \def\@@sectionbaselinestretch{\renewcommand*{\baselinestretch}{1.0}}
    \def\@@subsectionbaselinestretch{\renewcommand*{\baselinestretch}{1.0}}
    \def\@@subsectionseries{\normalfont}
    \def\@@subsubsectionseries{\normalfont}
    \def\@@thebibliographyparsep{\parsep 6pt}
  }
\ifthen{\equal{me}{\@@dept}}
  {
    \ifthenelse{\equal{\@@optionbibstyle}{jfm}}
      {\def\@@deptbibstyle{jfm}}
      {\ifthenelse{\equal{\@@optionbibstyle}{jfm2}}
        {\def\@@deptbibstyle{jfm2}}
        {\def\@@deptbibstyle{pumeunsrt}}
      }
    \def\fnum@table{\tablename~\thetable.~}
    \setboolean{@@figurecaptions}{true}
    \def\@@makechapterhead#1{\rm\thechapter. \uppercase{#1}}
    \def\@@makechapterheadspacea{\vspace*{0.5truein}}
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.5truein}}
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
    \nochapterblankpages
  }
\ifthen{\equal{ne}{\@@dept}}
  {
    \def\@@makechapterhead#1{
      \ifthenelse{\boolean{@@inappendix}}
        {\large\bf APPENDIX \thechapter\\\uppercase{#1}}
        {\large\bf\thechapter. \uppercase{#1}}
    }
    \def\@@makechapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
    \nochapterblankpages
  }
\ifthen{\equal{chem}{\@@dept}}
  {
    % The vertical spacing between floats appearing either at the
    % top or at the bottom of a page for double column floats in
    % two-column page format.
    \setlength{\dblfloatsep}{36pt}
    % The vertical spacing between floats and text, for both top
    % and bottom floats for double column floats in two-column page
    % format.
    \setlength{\dbltextfloatsep}{36pt}
    % The vertical spacing between floats appearing either at the
    % top or at the bottom of a page.
    \setlength{\floatsep}{36pt}
    % The vertical spacing above and below a float that appears in
    % the middle of a text page with the h placement argument.
    \setlength{\intextsep}{36pt}
    \def\@@makechapterhead#1{\thechapter. #1}
    \def\@@makechapterheadspacea{\vspace*{0.7truein}}
    \def\@@makechapterheadspaceb{\vspace*{36pt}}
    \def\@@makeschapterheadspacea{\vspace*{0.7truein}}
    \def\@@makeschapterheadspaceb{\vspace*{36pt}}
    % The vertical spacing between floats and text, for both top
    % and bottom floats.
    \setlength{\textfloatsep}{36pt}
  }
\ifthen{\equal{cs}{\@@dept}}
  {
    \def\@@makechapterhead#1{\thechapter\quad\uppercase{#1}}
    \def\@@makechapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.6truein}}  % by trial and error
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
    \def\@@sectionseries{}
    \def\@@subsectionseries{}
    \def\@@subsubsectionseries{}
  }
\ifthen{\equal{eas}{\@@dept}}
  {
    \def\@@makechapterhead#1{\thechapter\quad\uppercase{#1}}
    \def\@@makechapterheadspacea{\vspace*{0.5truein}}
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.5truein}}
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
  }
\ifthen{\equal{math}{\@@dept}}
  {
    \def\@@makechapterhead#1{\large\bf\thechapter. #1}
    \def\@@makechapterheadspacea{\vspace*{0.5truein}}
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.5truein}}
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
  }
\ifthen{\equal{phys}{\@@dept}}
  {
    \def\@@makechapterhead#1{\large\bf\thechapter. #1}
    \def\@@makechapterheadspacea{\vspace*{0.5truein}}
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.5truein}}
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
  }
\ifthen{\equal{stat}{\@@dept}}
  {
    \def\@@makechapterhead#1{\large\rm\thechapter. \uppercase{#1}}
    \def\@@makechapterheadspacea{\vspace*{0.5truein}}
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.5truein}}
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
  }
\AtBeginDocument{
  \usepackage{notoccite}
}
\AtEndDocument{
%%  \@@Identification
%%  \write\@auxout{\string\newcounter{\string\csname@@@appendix\endcsname}}
%%  \write\@auxout{\string\setcounter{\string\csname@@@appendix\endcsname}{\arabic{@@appendix}}}
}
\newenvironment{cland}
  {\begin{landscape}\hbox\bgroup\hss\vbox\bgroup}
  {\egroup\hss\egroup\end{landscape}}

\newenvironment{lquotation}
  {\begin{quotation}\renewcommand{\baselinestretch}{1}\reset@font}
  {\end{quotation}}

\newcommand{\coversheet}[1]{
  {
    \ifthen{\boolean{@@coversheets}}
      {
        \ifthenelse{\boolean{@@chapterblankpages}}
          {\clearpage} %% ! {\cleardoublepage}
          {\clearpage}
        \pagestyle{empty}
        \mbox{}
        \@@coversheetspace
        \begin{center}
          #1
        \end{center}
        \vfill
        \newpage
        \addtocounter{page}{-1}
      }
    \@@NotTableOfContents
  }
}
\setlength{\paperheight}{11truein}
\setlength{\paperwidth}{8.5truein}
\renewcommand*{\baselinestretch}{1.5}
\setlength{\evensidemargin}{0.5truein}
\setlength{\oddsidemargin}{0.5truein}
\setlength{\textheight}{8.75truein}
\setlength{\textwidth}{6truein}
\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000
\ifthen{\equal{mse}{\@@dept}}
  {\setlength{\parindent}{0.40625truein}}
%\setlength\parskip{0\p@ \@plus \p@}
\pagenumbering{roman}
\newcommand{\@@NotTableOfContents}{%
  \renewcommand*{\@evenhead}{\@@evenhead}
  \renewcommand*{\@oddhead}{\@@oddhead}
  \renewcommand*{\@evenfoot}{\@@evenfoot}
  \renewcommand*{\@oddfoot}{\@@oddfoot}
}
\@@NotTableOfContents
\setlength{\topmargin}{-0.5truein}
\addtolength{\topmargin}{0.04875truein}
\settoheight{\headheight}{- ivx1 -}
\setlength{\headsep}{0.5truein}
\addtolength{\headsep}{-\headheight}
\addtolength{\headsep}{-0.0625truein}
\newcommand*{\@@TitleAuthor}{\relax}
\newcommand*{\@@AbstractAuthor}{\relax}
\renewcommand*{\author}[2]{%
  \renewcommand*{\@@TitleAuthor}{#1}%
  \renewcommand*{\@@AbstractAuthor}{#2}%
}
\newcommand*{\@@Campus}{\relax}
\newcommand*{\@@Campus@Input}{}
\newcommand*{\campus}[1]{%
  \renewcommand\@@Campus@Input{#1}
  \renewcommand{\@@t}{Fort Wayne}      \ifthen{\equal{#1}{\@@t}}{\edef\@@Campus{\@@t}}
  \renewcommand{\@@t}{Hammond}         \ifthen{\equal{#1}{\@@t}}{\edef\@@Campus{\@@t}}
  \renewcommand{\@@t}{Indianapolis}    \ifthen{\equal{#1}{\@@t}}{\edef\@@Campus{\@@t}}
  \renewcommand{\@@t}{West Lafayette}  \ifthen{\equal{#1}{\@@t}}{\edef\@@Campus{\@@t}}
  \renewcommand{\@@t}{Westville}       \ifthen{\equal{#1}{\@@t}}{\edef\@@Campus{\@@t}}
}


\newcommand*{\@@TitleDegree}{\relax}
\newcommand*{\@@AbstractDegree}{\relax}
\newcommand*{\@@DegreeMonth}{\relax}
\newcommand*{\@@DegreeYear}{\relax}
\newcommand*{\pudegree}[4]{%
  \renewcommand*{\@@TitleDegree}{#1}%
  \renewcommand*{\@@AbstractDegree}{#2}%
  \renewcommand*{\@@DegreeMonth}{#3}%
  \renewcommand*{\@@DegreeYear}{#4}%
}
\newcommand*{\@@MajorProf}{\relax}
\newcommand*{\majorprof}[1]{\renewcommand*{\@@MajorProf}{Major Professor: #1}}
\newcommand*{\majorprofs}[1]{\renewcommand*{\@@MajorProf}{Major Professors: #1}}
\newcommand*{\@@Title}{\relax}
\renewcommand*{\title}[1]{\renewcommand*{\@@Title}{#1}}
\renewcommand*{\maketitle}
  {
    \ifthen{\equal{\@@Campus@Input}{}}
      {
        \IW{You must specify which campus your degree is from, for example,}
        \IW{\space\space\space\space\string\campus{West Lafayette}}
        \IW{See}
        \IW{\space\space\space\space http://engineering.purdue.edu/\string~mark/puthesis/\@@number campus}
        \IW{for the valid choices.}
        \IW{ABORTING...}
        \IW{}
        \stop
      }
    \ifthen{\equal{\@@Campus}{\relax}}
      {
        \IW{The campus specified in your}
        \IW{\space\space\space\space\string\campus{\@@Campus@Input}}
        \IW{command is not valid.}
        \IW{See}
        \IW{\space\space\space\space http://engineering.purdue.edu/\string~mark/puthesis/\@@number campus}
        \IW{for the valid choices.}
        \IW{ABORTING...}
        \IW{}
        \stop
      }
    {
      \renewcommand*{\baselinestretch}{2} \reset@font
      \setcounter{save@@page}{\value{page}}
      \begin{titlepage}
        \mbox{}
        \vfil
        \vfil
        \begin{center}
        \uppercase\expandafter{\@@Title}
        \end{center}
        \ifthen{\(\boolean{number@@all@@volumes} \and \value{last@@volume}>1\) \or \value{@@volume}>1}
          {
            \begin{center}
              VOLUME \the@@volume
            \end{center}
          }
        \vfil
        \begin{center}
          \ifthen{\equal{bypass}{\@@type}}{A Master's Bypass Report\\}
          \ifthen{\equal{dissertation}{\@@type}}{A Dissertation\\}
          \ifthen{\equal{prelim}{\@@type}}{A Preliminary Report\\}
          \ifthen{\equal{thesis}{\@@type}}{A Thesis\\}
          Submitted to the Faculty\\
          of\\
          Purdue University\\
          by\\
          \@@TitleAuthor
        \end{center}
        \vfil
        \begin{center}
          In Partial Fulfillment of the\\
          Requirements for the Degree\\
          of\\
          \@@TitleDegree
        \end{center}
        \ifthenelse{\equal{mse}{\@@dept}}
          {}
          {\vfil}
        \begin{center}
          \@@DegreeMonth\ \@@DegreeYear\\
          Purdue University\\
          \@@Campus, Indiana
        \end{center}
        \ifthenelse{\equal{mse}{\@@dept}}
          {}
          {\vfil\vfil}
      \end{titlepage}
      \setcounter{page}{\value{save@@page}}
      \ifthen{\value{@@volume}=0 \or \value{@@volume}=1}
        {\setcounter{page}{2}}
    }
  }
\newenvironment{statement}%
  {%
    \newpage
    \chapter*%
    {%
      THE PURDUE UNIVERSITY GRADUATE SCHOOL\\
      STATEMENT OF
      \ifthen{\equal{bypass}{\@@type}}{MASTER'S BYPASS}%
      \ifthen{\equal{dissertation}{\@@type}}{DISSERTATION}%
      \ifthen{\equal{prelim}{\@@type}}{PRELIMINARY REPORT}%
      \ifthen{\equal{thesis}{\@@type}}{THESIS}%
      ~APPROVAL%
    }
    \def\entry##1##2%
    {%
      \noindent##1\par
      \noindent\hspace*{0.5in}##2\par
    }
    \def\approvedby##1##2%
    {%
      \mbox{}\par
      \noindent{\bfseries Approved by:}\par
      \noindent\hspace*{0.5in}##1\par
      \noindent\hspace*{1in}##2\par
    }
    \mbox{}\par
  }
  {%
  }
  
\newenvironment{dedication}%
  {%
    \newpage
    \mbox{}
    \vfil
    \begin{center}%
  }%
  {%
    \end{center}%
    \vfil
    \eject
    \@@NotTableOfContents
  }
\newboolean{@@inother}
\setboolean{@@inother}{false}
  % #1 "next" or "odd": start on next or next odd page?
  % #2 what to print at top of page
  % #3 "y" or "n": put in table of contents?
  % #4 amount of extra space to put after heading at top of page
\newcommand{\@@nonchapter}[4]{{%
  \@@NotTableOfContents
  \bgroup
    \setboolean{@@inother}{true}
    \renewcommand{\large}{}%
    \renewcommand{\bf}{}%
    \ifthenelse{\equal{ce}{\@@dept}}
      {\chapter*{\uppercase{#2}}}
      {\chapter*{#2}}
    \ifthen{\equal{y}{#3}}
      {
        \ifthenelse{\equal{ce}{\@@dept}}
          {\addcontentsline{toc}{chapter}{\uppercase{#2}}}
          {\addcontentsline{toc}{chapter}{#2}}
      }
  \egroup
  \vspace{#4}
}}
\newenvironment{acknowledgments}%
  {\@@nonchapter{next}{ACKNOWLEDGMENTS}{n}{0pt}}%
  {}
\newenvironment{preface}%
  {\@@nonchapter{next}{PREFACE}{n}{0pt}}%
  {}
\renewcommand*{\tableofcontents}{
  \@@nonchapter{odd}{TABLE OF CONTENTS}{n}{0pt}
  {\leftskip=0pt \noindent\hbox to\textwidth{\hfil Page}\par}
  {%
%   \renewcommand*{\@oddhead}{%
%     \hfil\textrm{\hfil oddhead a \thepage}%
%     \renewcommand*{\@oddhead}{\hfil oddhead b \thepage}%
%   }
%   \renewcommand*{\@evenhead}{%
%     \hfil\textrm{\hfil evenhead a \thepage}%
%     \renewcommand*{\@evenhead}{\hfil evenhead b \thepage}%
%   }
    \output={
      \let \par \@@par
      \ifnum \outputpenalty<-\@M
        \@specialoutput
      \else
        \@makecol
        \@opcol
        \@startcolumn
        \@whilesw \if@fcolmade \fi
          {%
           \@opcol\@startcolumn}%
      \fi
      \ifnum \outputpenalty>-\@Miv
        \ifdim \@colroom<1.5\baselineskip
          \ifdim \@colroom<\textheight
            \@latex@warning@no@line{Text page \thepage\space
              contains only floats}%
            \@emptycol
          \else
            \global \vsize \@colroom
          \fi
        \else
          \global \vsize \@colroom
        \fi
      \else
        \global \vsize \maxdimen
      \fi
      {\leftskip=0pt \noindent\hbox to\textwidth{\hfil Page}\par}
    }
    \renewcommand{\baselinestretch}{1}\reset@font
    \@starttoc{toc}
  }
}
\ifthen{\equal{che}{\@@dept}\or \equal{ece}{\@@dept}\or \equal{hsci}{\@@dept}\or \equal{me}{\@@dept}}
  {%
    \def\numberline#1{\hb@xt@\@tempdima{#1\hfil}}
    \renewcommand*\l@figure{\@dottedtocline{1}{\bigtocskip}{0em}{2.3em}}
    \renewcommand*\l@table{\@dottedtocline{1}{\bigtocskip}{0em}{2.3em}}
  }
\def\bigtocskip{0.5\baselineskip plus.2\p@}
\def\smalltocskip{0pt}
\def\@dottedtocline#1#2#3#4#5#6{%
  \ifnum #1>\c@tocdepth
  \else
    \vskip #2
    {%
      \leftskip #3
      \rightskip \@tocrmarg
      \parfillskip -\rightskip
      \parindent #3
      \@afterindenttrue
      \interlinepenalty\@M
      \leavevmode
      \@tempdima #4
      \advance\@tempdima \@@padding
      \advance\leftskip \@tempdima
      \hbox{}\hskip -\leftskip
      #5\nobreak
      \leaders\hbox{$\m@th \mkern \@dotsep mu.\mkern \@dotsep mu$}\hfill
      \nobreak
      \renewcommand{\@pnumwidth}{1em}%
      \setlength{\@@tlength}{\@pnumwidth}%
      \settowidth{\@@ulength}{\reset@font \rm #6}%
      \ifdim \@@ulength>\@@tlength
        \hbox to\@@ulength{\hfil\reset@font \rm #6}\par
      \else
        \hbox to\@@tlength{\hfil\reset@font \rm #6}\par
      \fi
    }%
  \fi
}
\renewcommand*{\listoftables}{
  \@@nonchapter{next}{LIST OF TABLES}{y}{0pt}
  {\leftskip=0pt \noindent\hbox to\textwidth{Table\hfil Page}\par}
  {%
    \output={
      \let \par \@@par
      \ifnum \outputpenalty<-\@M
        \@specialoutput
      \else
        \@makecol
        \@opcol
        \@startcolumn
        \@whilesw \if@fcolmade \fi
          {%
           \@opcol\@startcolumn}%
      \fi
      \ifnum \outputpenalty>-\@Miv
        \ifdim \@colroom<1.5\baselineskip
          \ifdim \@colroom<\textheight
            \@latex@warning@no@line{Text page \thepage\space
              contains only floats}%
            \@emptycol
          \else
            \global \vsize \@colroom
          \fi
        \else
          \global \vsize \@colroom
        \fi
      \else
        \global \vsize \maxdimen
      \fi
      {\leftskip=0pt \noindent\hbox to\textwidth{Table\hfil Page}\par}
    }
    \renewcommand{\baselinestretch}{1}\reset@font
    \@starttoc{lot}
  }
}
\renewcommand{\listoffigures}{
  \@@nonchapter{next}{LIST OF FIGURES}{y}{0pt}
  {\leftskip=0pt \noindent\hbox to\textwidth{Figure\hfil Page}\par}
  {%
    \output={
      \let \par \@@par
      \ifnum \outputpenalty<-\@M
        \@specialoutput
      \else
        \@makecol
        \@opcol
        \@startcolumn
        \@whilesw \if@fcolmade \fi
          {%
           \@opcol\@startcolumn}%
      \fi
      \ifnum \outputpenalty>-\@Miv
        \ifdim \@colroom<1.5\baselineskip
          \ifdim \@colroom<\textheight
            \@latex@warning@no@line{Text page \thepage\space
              contains only floats}%
            \@emptycol
          \else
            \global \vsize \@colroom
          \fi
        \else
          \global \vsize \@colroom
        \fi
      \else
        \global \vsize \maxdimen
      \fi
      {\leftskip=0pt \noindent\hbox to\textwidth{Figure\hfil Page}\par}
    }
    \renewcommand{\baselinestretch}{1}\reset@font
    \@starttoc{lof}
  }
}
\newcommand{\@@startlist}[1]{
  \@@nonchapter{odd}{#1}{y}{0pt}%
  \setlength{\LTleft}{\parindent}%
  \setlength{\LTright}{0truein}%
  %% ! change to \setlength[\@@tlength}{\textwidth - \LTleft - \LTright - 2*\tabcolsep - 1truein} later
  \setlength{\@@tlength}{\textwidth}
  \addtolength{\@@tlength}{-\LTleft}
  \addtolength{\@@tlength}{-\LTright}
  \addtolength{\@@tlength}{-\tabcolsep}
  \addtolength{\@@tlength}{-\tabcolsep}
  \addtolength{\@@tlength}{-1truein}
    %% ! Get real width of first column.
  %% ! \setcounter{\@@tcount}{\c@LT@tables}
  %% ! \addtocounter(\@@tcount){1}
  %% ! \def\stuff{\csname LT@\romannumeral\@@tcount\endcsname}
  %% ! \showthe\stuff
  %% ! \addtolength{\@@tlength}{-?}
}
\newenvironment{symbols}%
  {%
    \@@startlist{SYMBOLS}
    \begin{longtable}{lp{\@@tlength}}%
  }
  {\end{longtable}}
\newenvironment{abbreviations}
  {
    \@@startlist{ABBREVIATIONS}
    \begin{longtable}{lp{\@@tlength}}%
  }
  {\end{longtable}}
\newenvironment{nomenclature}
  {
    \@@startlist{NOMENCLATURE}
    \begin{longtable}{lp{\@@tlength}}%
  }
  {\end{longtable}}
\renewenvironment{glossary}
  {
    \@@startlist{GLOSSARY}
    \begin{longtable}{lp{\@@tlength}}%
  }
  {\end{longtable}}
\renewenvironment{abstract}%
  {%
    \@@nonchapter{next}{ABSTRACT}{y}{0pt}
    {%
      \renewcommand*{\baselinestretch}{1.0}\reset@font
      \vspace*{\baselineskip}
      \vbox{
        \noindent
        \@@AbstractAuthor~\@@AbstractDegree,
        Purdue University,
        \@@DegreeMonth\ %
        \@@DegreeYear.
        {%
          \renewcommand*{\\}{}%
          \@@Title.
        }
        \ifthenelse{\isundefined{\@@OverrideMajorProf}}
          {\@@MajorProf\ifthen{\equal{me}{\@@dept}}{, School of Mechanical Engineering}.}
          {\@@OverrideMajorProf}
      }
      \vspace*{\baselineskip}
      \par
    }
  }%
  {\par}
\renewcommand{\chapter}{%
  \ifthenelse{
        \(      \boolean{@@inother} \and \boolean{@@nonchapterblankpages} \)
    \or \( \not \boolean{@@inother} \and \boolean{@@chapterblankpages}    \)
  }
    {\cleardoublepage}
    {\clearpage}
  \@@NotTableOfContents
  \global\@topnum\z@
  \@afterindentfalse
  \secdef \@chapter \@schapter
}
\newboolean{@@inchapters}
\setboolean{@@inchapters}{false}
\def\@chapter[#1]#2{%
  \ifnum \c@secnumdepth >\m@ne
    \refstepcounter{chapter}%
    \typeout{\@chapapp\space\thechapter.}%
    \setboolean{@@unset}{true}
    \ifthen{\equal{agecon}{\@@dept}}
      {\setboolean{@@unset}{false}\addcontentsline{toc}{chapter}{\protect\uppercase{\@chapapp\ \thechapter. #1}}}
    \ifthen{\equal{ce}{\@@dept}}
      {\setboolean{@@unset}{false}\addcontentsline{toc}{chapter}{\protect\numberline{\thechapter}\uppercase{#1}}}
    \ifthen{\equal{me}{\@@dept}}
      {\setboolean{@@unset}{false}\addcontentsline{toc}{chapter}{\protect\numberline{\thechapter.}\uppercase{#1}}}
    \ifthen{\equal{mse}{\@@dept}}
      {\setboolean{@@unset}{false}\addcontentsline{toc}{chapter}{\protect\uppercase{\@chapapp\ \thechapter. #1}}}
    \ifthen{\boolean{@@unset}}
      {\addcontentsline{toc}{chapter}{\protect\numberline{\thechapter}#1}}
  \else
    \ifthenelse{\equal{ce}{\@@dept}}
      {\addcontentsline{toc}{chapter}{\uppercase{#1}}}
      {\addcontentsline{toc}{chapter}{#1}}
  \fi
  \chaptermark{#1}%
% \addtocontents{lof}{\protect\addvspace{10\p@}}%
% \addtocontents{lot}{\protect\addvspace{10\p@}}%
  \ifthenelse{\equal{ce}{\@@dept}}
    {\@makechapterhead{\uppercase{#2}}}
    {\@makechapterhead{#2}}
  \@afterheading
  \ifthen{\not \boolean{@@inchapters}}
    {
      \pagenumbering{arabic}%
      \@@inchapterstrue
    }
}
\def\@makechapterhead#1{%
  \@@makechapterheadspacea
  {\centering
    \@@makechapterhead{#1}
    \endgraf
  }
  \@@makechapterheadspaceb
}
\def\@makeschapterhead#1{%
  \@@makeschapterheadspacea
  {\centering
    \ifthenelse{\equal{ce}{\@@dept}}
      {\large\bf\uppercase{#1}}
      {\large\bf #1}
    \endgraf
  }
  \@@makeschapterheadspaceb
}
\def\@startsection#1#2#3#4#5#6{%
  \if@noskipsec
    \leavevmode
  \fi
  \par
  \@tempskipa #4\relax
  \@afterindenttrue
  \ifdim \@tempskipa <\z@
    \@tempskipa -\@tempskipa
  \fi
  \if@nobreak
    \everypar{}%
  \else
    \addpenalty{\@secpenalty}%
    \addvspace{\@tempskipa}%
  \fi
  \@ifstar
    {\@ssect{#3}{#4}{#5}{#6}}%
    {\@dblarg{\@sect{#1}{#2}{#3}{#4}{#5}{#6}}}%
}
\newenvironment{cabstract}{\begin{quote}\textbf{Abstract}\quad}{\end{quote}}
\newtheorem{definition}{Definition}[section]
\newtheorem{observation}{Observation}[section]
\newsavebox{\proofbox}
\sbox{\proofbox}{\rule{7pt}{7pt}}
\newtheorem{Proof}{Proof}
\renewcommand{\theProof}{}
\newenvironment{proof}{\begin{Proof}\rm}{\hfill \usebox{\proofbox} \end{Proof}}
\newtheorem{proposition}{Proposition}[section]
\newtheorem{theorem}{Theorem}[section]
\renewcommand{\section}{%
  \@startsection
    {section}%
    {1}%
    {\z@}%
    {24pt}%
    {12pt}%
    {\@@sectionbaselinestretch\reset@font\@@sectionseries}%
}
\ifthen{\boolean{@@uglyheadings}}
  {
    \def\undertext#1{$\underline{\smash{\hbox{#1}}}$}
    \def\section{
      \vspace{36truept}
      \secdef{\@@section}{\@@ssection}
    }
    \def\@@section[#1]#2{
      \refstepcounter{section}
      \sectionmark{#1}
      \hbox to \textwidth{\hss\undertext{\thesection\ #2}\hss}\nopagebreak
      \addcontentsline{toc}{section}{\protect\numberline{\csname thesection\endcsname}#1}
      \nopagebreak
    }
    \def\@@ssection#1{
      \sectionmark{#1}
      \hbox to \textwidth{\hss\undertext{#1}\hss}\nopagebreak
      \nopagebreak
    }
  }
\renewcommand{\subsection}{%
  \@startsection
    {subsection}%
    {2}%
    {\z@}%
    {24pt}% \@plus -1ex \@minus -.2ex}%
    {12pt}% 1.5ex \@plus .2ex}%
    {\@@subsectionbaselinestretch\reset@font\@@subsectionseries}%
}
\ifthen{\boolean{@@uglyheadings}}
  {
    \renewcommand{\subsection}[1]{
      \refstepcounter{subsection}
      \vspace{36truept}
      \subsectionmark{#1}
      \hbox to \textwidth{\hss\thesubsection\ #1\hss}\nopagebreak
      \addcontentsline{toc}{subsection}{\protect\numberline{\csname thesubsection\endcsname}#1}
      \nopagebreak
    }
  }
\renewcommand{\subsubsection}{
  \@startsection
    {subsubsection}%
    {3}%
    {\z@}%
    {24pt}% \@plus -1ex \@minus -.2ex}%
    {12pt}% 1.5ex \@plus .2ex}%
    {\@@subsubsectionbaselinestretch\reset@font\@@subsubsectionseries}%
}
\ifthen{\equal{mse}{\@@dept}}
  {
    \def\undertext#1{$\underline{\hbox{#1}}$}
    \newbox{\@@Strut}
    \renewcommand{\subsubsection}[1]{
      \refstepcounter{subsubsection}
      \vspace{6truept}
      \subsubsectionmark{#1}
      \noindent \undertext{\vphantom{gjpqy}\thesubsubsection. #1}\newline\nopagebreak
      \addcontentsline{toc}{subsubsection}{\protect\numberline{\csname thesubsubsection\endcsname}#1}%
      \nopagebreak\indent
    }
  }
\ifthen{\boolean{@@uglyheadings}}
  {
    \renewcommand{\subsubsection}[1]{
      \refstepcounter{subsubsection}
      \vspace{36truept}
      \subsubsectionmark{#1}
      \noindent \undertext{\thesubsubsection\ #1}\newline\nopagebreak
      \addcontentsline{toc}{subsubsection}{\protect\numberline{\csname thesubsubsection\endcsname}#1}
      \nopagebreak
    }
  }
\ifthen{\boolean{@@uglyheadings}}
  {
    \setcounter{tocdepth}{4}
    \newcommand*{\l@subsubsubsection}{\@dottedtocline{4}{\smalltocskip}{11.2em}{5.0em}}
    \def\subsubsubsectionmark#1{}
    \newcounter{subsubsubsection}[subsubsection]
    \renewcommand{\thesubsubsubsection}{\thesubsubsection.\arabic{subsubsubsection}}
    \newcommand{\subsubsubsection}[1]{
      \refstepcounter{subsubsubsection}
      \vspace{36truept}
      \subsubsubsectionmark{#1}
      \noindent \thesubsubsubsection. #1\newline\nopagebreak
      \addcontentsline{toc}{subsubsubsection}{\protect\numberline{\csname thesubsubsubsection\endcsname}#1}
      \nopagebreak
    }
  }
\def\@sect#1#2#3#4#5#6[#7]#8{%
  \ifnum #2>\c@secnumdepth
    \let\@svsec\@empty
  \else
    \refstepcounter{#1}%
    \edef\@svsec{\csname the#1\endcsname\hskip 1em}%
  \fi
  \ifthen{\equal{che}{\@@dept}}
    {%
      \ifnum #2=3
        \refstepcounter{#1}%
        \edef\@svsec{\csname the#1\endcsname\hskip 1em}%
      \fi
    }%
  \@tempskipa #5\relax
  \ifdim \@tempskipa>\z@
    \begingroup
      #6\relax
      \@hangfrom{\hskip #3\relax\@svsec}%
      {\interlinepenalty \@M #8\par}%
    \endgroup
    \csname #1mark\endcsname{#7}%
    \addcontentsline{toc}{#1}{%
      \ifnum #2>\c@secnumdepth
      \else
        \protect\numberline{\csname the#1\endcsname}%
      \fi
      #7%
    }%
  \else
    \def\@svsechd{%
      #6%
      \hskip #3\relax
      \@svsec #8\csname #1mark\endcsname
      {#7}%
      \addcontentsline{toc}{#1}{%
        \ifnum #2>\c@secnumdepth
        \else
          \protect\numberline{\csname the#1\endcsname}
        \fi
        #7%
      }%
    }%
  \fi
  \@xsect{#5}%
}
\renewcommand*{\l@chapter}{\@dottedtocline{0}{\bigtocskip}{0em}{1.4em}}
\ifthen{\equal{ece}{\@@dept}\or \equal{hsci}{\@@dept}}
  {\renewcommand{\figurename}{Fig.}}
\renewcommand{\fnum@figure}{\figurename~\thefigure.}
\renewcommand{\fnum@table}{\tablename~\thetable.}
\newboolean{@@centercaption}
\renewcommand{\caption}{%
  \setboolean{@@centercaption}{true}
  \refstepcounter\@captype \@dblarg{\@caption\@captype}%
}
\newcommand\bcaption{%
  \setboolean{@@centercaption}{true}
  \refstepcounter\@captype \@dblarg{\@caption\@captype}%
}
\long\def\@makecaption#1#2{%
  \vspace*{\abovecaptionskip}
  \ifthenelse{\boolean{@@centercaption}}  % center caption
    {
      \setlength{\@@captionwidth}{\textwidth}
      \addtolength{\@@captionwidth}{-4\@@parindent}
      \ifthenelse{\equal{figure}{\@captype} \or \boolean{@@figurecaptions}}
        {
          \sbox\@tempboxa{#1 #2}%
          \renewcommand{\baselinestretch}{1.0}\reset@font
          \ifdim \wd\@tempboxa >\hsize
            \centerline{\parbox[t]{\@@captionwidth}{#1 #2}}
          \else
            \centerline{#1 #2}%
          \fi
        }
        {
          \renewcommand{\baselinestretch}{1.0}\reset@font
          \centerline{#1}
          \sbox\@tempboxa{#2}%
          \ifdim \wd\@tempboxa >\hsize
            \centerline{\parbox[t]{\@@captionwidth}{#2}}
          \else
            \centerline{#2}%
          \fi
        }
      \vspace*{\belowcaptionskip}
    }
    {
      \setlength{\topsep}{0pt}
      \setlength{\parskip}{0pt}
      \setlength{\partopsep}{0pt}
      \begin{quote}
        \renewcommand{\\}{}
        \renewcommand{\baselinestretch}{1.0}\reset@font
        \ifthenelse{\equal{aae}{\@@dept}}
          {#1. #2\par}
          {
            \ifthenelse{\equal{table}{\@captype}}
              {\hfil\strut #1\hfil\break #2\par}
              {#1 #2\par}
          }
      \end{quote}
    }
  \vskip\belowcaptionskip
}
\setlength\belowcaptionskip{5\p@}
\long\def\@makefntext#1{%
  \baselineskip=12pt
  \noindent
  \@makefnmark #1%
}
\ifthenelse{\equal{}{\@@optionbibstyle}}
  {\newcommand{\@@bibstyle}{\@@deptbibstyle}}
  {\newcommand{\@@bibstyle}{\@@optionbibstyle}}

\ifthen{\equal{abbrv}{\@@bibstyle}}
  {
    \renewcommand{\bibname}{{\normalsize\rm REFERENCES}}
    \bibliographystyle{abbrv}
  }

\ifthen{\equal{abbrvnat}{\@@bibstyle}}
  {
    \usepackage[sort]{natbib}
    \renewcommand{\bibname}{{\normalsize\rm REFERENCES}}
    \bibliographystyle{abbrvnat}
  }

\ifthen{\equal{aer}{\@@bibstyle}}
  {
    \usepackage{harvard}
    \bibliographystyle{aer}
  }


\ifthen{\equal{agsm}{\@@bibstyle}}
  {
    \usepackage{cite}
    \usepackage{harvard}
    \bibliographystyle{agsm}
  }

\ifthen{\equal{aip}{\@@bibstyle}}
  {
    \usepackage{cite}
    \usepackage{revtex}
    \renewcommand{\bibname}{{\normalsize\rm REFERENCES}}
    \bibliographystyle{aip}
  }

\ifthen{\equal{alpha}{\@@bibstyle}}
  {
    \usepackage{cite}
    \bibliographystyle{alpha}
  }

\ifthen{\equal{ama}{\@@bibstyle}}
{
    \usepackage{cite}
    \bibliographystyle{ama}
}

\ifthen{\equal{apacite}{\@@bibstyle}}
  {
    \usepackage{apacite}
    \bibliographystyle{apacite}
  }

\ifthen{\equal{apalike}{\@@bibstyle}}
  {
    \usepackage{apalike}
    \bibliographystyle{apalike}
  }

\ifthen{\equal{astron}{\@@bibstyle}}
  {
    \usepackage{astron}
    \bibliographystyle{astron}
  }

\ifthen{\equal{chicago}{\@@bibstyle}}
  {
    \usepackage{natbib}
    \bibliographystyle{chicago}
  }

\ifthen{\equal{ieee}{\@@bibstyle}}
  {
    \usepackage{cite}
    \bibliographystyle{ieee}
  }

\ifthen{\equal{IEEEtran}{\@@bibstyle}}
  {
    \usepackage{cite}
    \bibliographystyle{IEEEtran}
  }

\ifthen{\equal{jfm}{\@@bibstyle}}
  {
    \usepackage{natbib}
    \bibliographystyle{jfm}
  }

\ifthen{\equal{jfm2}{\@@bibstyle}}
  {
    \usepackage{natbib}
    \bibliographystyle{jfm2}
  }

\ifthen{\equal{kluwer}{\@@bibstyle}}
  {
    \usepackage{harvard}
    \bibliographystyle{kluwer}
  }

\ifthen{\equal{phaip}{\@@bibstyle}}
  {
    \bibliographystyle{phaip}
  }

\ifthen{\equal{pumeunsrt}{\@@bibstyle}}
  {
    \usepackage{cite}
    \bibliographystyle{pumeunsrt}
  }

\ifthen{\equal{plain}{\@@bibstyle}}
  {
    \usepackage{cite}
    \bibliographystyle{plain}
  }

\ifthen{\equal{plainnat}{\@@bibstyle}}
  {
    \usepackage{natbib}
    \renewcommand{\bibname}{{\normalsize\rm REFERENCES}}
    \bibliographystyle{plainnat}
  }

\ifthen{\equal{unified}{\@@bibstyle}}
{
    \usepackage{natbib}
    \bibliographystyle{unified}
    \def\@@bibname{{\normalsize\rm REFERENCES}}
    \nochapterblankpages
}

\ifthen{\equal{unsrt}{\@@bibstyle}}
  {
    \usepackage{cite}
    \bibliographystyle{unsrt}
  }

\ifthen{\equal{unsrtnat}{\@@bibstyle}}
  {
    \usepackage{natbib}
    \renewcommand{\bibname}{{\normalsize\rm REFERENCES}}
    \bibliographystyle{unsrtnat}
  }

\renewenvironment{thebibliography}[1]
  {
    \@@startthebibliography
    \@@nonchapter{odd}{\@@bibname}{y}{24pt}
    \list
      {\@biblabel{\arabic{enumiv}}}%
      {%
        \settowidth\labelwidth{\@biblabel{#1}}%
        \leftmargin\labelwidth
        \advance\leftmargin\labelsep
        \if@openbib
          \advance\leftmargin\bibindent
          \itemindent -\bibindent
          \listparindent \itemindent
          \parsep \z@
        \fi
        \@@thebibliographyparsep
        \usecounter{enumiv}%
        \let\p@enumiv\@empty
        \renewcommand{\theenumiv}{\arabic{enumiv}}%
        \baselineskip=12pt
      }%
    \if@openbib
      \renewcommand{\newblock}{\par}
    \else
      \renewcommand{\newblock}{\hskip .11em \@plus.33em \@minus.07em}%
    \fi
    \sloppy\clubpenalty4000\widowpenalty4000%
    \sfcode`\.=\@m
  }
  {}
\newboolean{@@inappendix}
\newcommand{\@@appendixname}{Appendix}
\renewcommand{\appendix}{\par
  \setboolean{@@inappendix}{true}
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \coversheet{\uppercase\expandafter{\@@appendixname}}
  \renewcommand{\@chapapp}{\appendixname}%
  \renewcommand{\thechapter}{\Alph{chapter}}
}
\newcommand{\@@appendicesname}{Appendices}
\newcommand{\appendices}{\par
  \setboolean{@@inappendix}{true}
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \coversheet{\uppercase\expandafter{\@@appendicesname}}
  \renewcommand{\@chapapp}{\appendixname}%
  \renewcommand{\thechapter}{\Alph{chapter}}
}
\newcommand{\@@vitaname}{VITA}
\newenvironment{vita}
  {
    \@@startvita
    \@@nonchapter{odd}{\@@vitaname}{y}{0pt}%
  }
  {}
\def\@starttoc#1{%
  \begingroup
    \@input{\jobname.#1}%
    \if@filesw \expandafter\newwrite\csname tf@#1\endcsname
      \immediate\openout \csname tf@#1\endcsname \jobname.#1\relax
    \fi
    \global\@nobreakfalse
  \endgroup
}

\setcounter{topnumber}{10}
\renewcommand{\topfraction}{1}
\setcounter{bottomnumber}{10}
\renewcommand{\bottomfraction}{1}
\setcounter{totalnumber}{10}
\renewcommand{\textfraction}{0}
\renewcommand{\floatpagefraction}{0}

\setlength{\floatsep}{18pt plus 3pt minus 3pt}
\setlength{\textfloatsep}{30pt plus 3pt minus 6pt}
\setlength{\intextsep}{18pt plus 3pt minus 3pt}
\setlength{\dblfloatsep}{18pt plus 3pt minus 3pt}
\setlength{\dbltextfloatsep}{30pt plus 3pt minus 6pt}

\newcommand{\Baselinestretch}[1]{\renewcommand{\baselinestretch}{#1}\reset@font}

\def\volume{
  \addtocounter{@@volume}{1}
  \write\@auxout{\string\setcounter{last@@volume}{\the@@volume}}
  \ifthen{\(\boolean{number@@all@@volumes} \and \value{last@@volume}>1\) \or \value{@@volume}>1}
    {
      \addtocontents{toc}{\string\vskip\string\baselineskip \noindent VOLUME \the@@volume}
    }
  \maketitle
}

\ifthen{\equal{me}{\@@dept}}
  {
    \ifthenelse{\equal{\@@optionbibstyle}{jfm}}
      {\def\@@deptbibstyle{jfm}}
      {\ifthenelse{\equal{\@@optionbibstyle}{jfm2}}
        {\def\@@deptbibstyle{jfm2}}
        {\def\@@deptbibstyle{pumeunsrt}}
      }
    \def\fnum@figure{\figurename~\thefigure.~}
    \def\fnum@table{\tablename~\thetable.~}
    \setboolean{@@figurecaptions}{true}
    \def\@@makechapterhead#1{\rm\thechapter. \uppercase{#1}}
    \def\@@makechapterheadspacea{\vspace*{0.5truein}}
    \def\@@makechapterheadspaceb{\vspace{0.5\baselineskip}}
    \def\@@makeschapterheadspacea{\vspace*{0.5truein}}
    \def\@@makeschapterheadspaceb{\vspace*{0.5\baselineskip}}
    \nochapterblankpages
  }

\makeatother

\raggedbottom
