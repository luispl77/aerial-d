% #############################################################################
%% This is file `istulthesis.cls' file for the Thesis-MSc
%%
%% This Class is based on standard LaTeX class 'report.cls'
%%
% #############################################################################
% Thesis-MSc
% Version 4.1, January 2023
% BY: Prof. Rui Santos Cruz, rui.s.cruz@tecnico.ulisboa.pt
% #############################################################################
%
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{istulthesis}[2021/02/10]
%
\typeout{}
\typeout{--------------------------------------------------------------}
\typeout{ +---+ Thesis Style (istulthesis.cls)                         }
\typeout{ +---+ Version 3.1, February 2021                               }
\typeout{ +---+  for Instituto Superior Tecnico (IST)                  }
\typeout{ +---+  Universidade de Lisboa                                }
\typeout{ * Modified from 'report.cls'                                 }
\typeout{ * Created to write Dissertations                             }
\typeout{ * Conforms with IST Master Degree format                     }
\typeout{                                                              }
\typeout{ AUTHOR: Rui Santos Cruz                                      }
\typeout{ INSPIRED BY: Pedro Tomás work                                }
\typeout{--------------------------------------------------------------}
\typeout{}
% -----------------------------------------------------------------------------
% Certain options are not handled here but are passed on to report.cls
% by \PassOptionToClass.
%
\DeclareOption{fleqn}{ \PassOptionsToClass{fleqn}{report} }
\DeclareOption{leqno}{ \PassOptionsToClass{leqno}{report} }
\DeclareOption{openbib}{ \PassOptionsToClass{openbib}{report} }
\DeclareOption{draft}{ \PassOptionsToClass{draft}{report} }
\DeclareOption{final}{ \PassOptionsToClass{final}{report} }
\DeclareOption{openright}{ \PassOptionsToClass{openright}{report} }
\DeclareOption{openany}{ \PassOptionsToClass{openany}{report} }
\DeclareOption{10pt}{ \PassOptionsToClass{10pt}{report} }
\DeclareOption{11pt}{ \PassOptionsToClass{11pt}{report} }
\DeclareOption{12pt}{ \PassOptionsToClass{12pt}{report} }
\DeclareOption{oneside}{ \PassOptionsToClass{oneside}{report} }
\DeclareOption{twoside}{ \PassOptionsToClass{twoside}{report} }
\DeclareOption{a4paper}{ \PassOptionsToClass{a4paper}{report} }
\DeclareOption{a5paper}{ \PassOptionsToClass{a5paper}{report} }
\DeclareOption{b5paper}{ \PassOptionsToClass{b5paper}{report} }
\DeclareOption{letterpaper}{ \PassOptionsToClass{letterpaper}{report} }
\DeclareOption{executivepaper}{ \PassOptionsToClass{executivepaper}{report} }
\DeclareOption{letterpaper}{ \PassOptionsToClass{letterpaper}{report} }
\DeclareOption{letterpaper}{ \PassOptionsToClass{letterpaper}{report} }
\DeclareOption{letterpaper}{ \PassOptionsToClass{letterpaper}{report} }
% -----------------------------------------------------------------------------
% Invalid options - normally used in report.cls and others, but not here:
%
\DeclareOption{titlepage}
   {\ClassWarning{istulthesis}{Title page always generated. Option ignored.}}
\DeclareOption{notitlepage}
   {\ClassWarning{istulthesis}{Option 'notitlepage' not available for thesis.}}
\DeclareOption{twocolumn}
   {\ClassWarning{istulthesis}{Option 'twocolumn' not available for thesis.}}
\DeclareOption{landscape}
   {\ClassWarning{istulthesis}{Option 'landscape' not available for thesis.}}
\DeclareOption{onecolumn}
   {\ClassWarning{istulthesis}{Option 'onecolumn' superfluous.  Ignored.}}
%
% -----------------------------------------------------------------------------
% CUSTOM OPTIONS:
% Options for no decimal numbering, or decimal number nested over 3 deep:
\newif\if@AllHeadings
\newif\if@NumberStdHeadings
\newif\if@PerChptrNumbering
\newif\if@emphisbold
\newif\if@boldhead
\newif\if@fixedHeadSz
\newif\if@headSizeTwelvePt
\newif\if@ChapterDefault
\newif\if@SectionDefault
\newif\if@SubSectionDefault
\newif\if@OtherHeadingDefault
\newif\if@centerChapter
\newif\if@centerSection
\newif\if@inlineSubSubSection
\newif\if@numberSubSubSection
\newif\if@chapterCaps
\@AllHeadingsfalse
\@NumberStdHeadingstrue
\@PerChptrNumberingtrue
\@emphisboldtrue
\@boldheadtrue
\@fixedHeadSzfalse
\@centerChaptertrue
\@centerSectionfalse
\@inlineSubSubSectionfalse
\@numberSubSubSectiontrue
\@chapterCapsfalse
\@headSizeTwelvePttrue
\DeclareOption{consecutivenumbering}{ \@PerChptrNumberingfalse }
\DeclareOption{inlineh4}{ \@inlineSubSubSectiontrue }
\DeclareOption{nonumh4}{ \@numberSubSubSectionfalse }
\DeclareOption{centerh1}{ \@centerSectiontrue }
\DeclareOption{chapleft}{ \@centerSectiontrue }
\DeclareOption{chapup}{ \@chapterCapstrue }
\DeclareOption{emph2underline}{ \@emphisboldfalse }
\DeclareOption{plainhead}{ \@boldheadfalse }
\DeclareOption{fixedheadsize}{ \@fixedHeadSztrue }
\DeclareOption{varheadsize}{\@headSizeTwelvePtfalse}% Hidden option.  Don't use.
\DeclareOption{numberchapteronly}{
	\@PerChptrNumberingfalse\@NumberStdHeadingsfalse\@AllHeadingsfalse
	\@centerChaptertrue\@centerSectiontrue
	\@inlineSubSubSectiontrue\@numberSubSubSectionfalse }
\DeclareOption{allheadings}{
	\@PerChptrNumberingtrue\@NumberStdHeadingstrue\@AllHeadingstrue
	\@centerChaptertrue\@centerSectionfalse
	\@inlineSubSubSectionfalse\@numberSubSubSectiontrue}
\DeclareOption{typewriterstyle}{ \@emphisboldfalse
	\@PerChptrNumberingfalse\@NumberStdHeadingsfalse\@AllHeadingsfalse
	\@centerChaptertrue\@centerSectiontrue\@chapterCapstrue
	\@inlineSubSubSectiontrue\@numberSubSubSectionfalse
	\@boldheadfalse\@fixedHeadSztrue}
\DeclareOption{modernstyle}{ \@emphisboldtrue
	\@PerChptrNumberingtrue\@NumberStdHeadingstrue\@AllHeadingsfalse
	\@centerChaptertrue\@centerSectionfalse\@chapterCapsfalse
	\@inlineSubSubSectiontrue\@numberSubSubSectionfalse
	\@boldheadtrue\@fixedHeadSzfalse}
\DeclareOption{defaultstyle}{ \@emphisboldtrue
	\@PerChptrNumberingtrue\@NumberStdHeadingstrue\@AllHeadingsfalse
	\@centerChaptertrue\@centerSectionfalse\@chapterCapsfalse
	\@inlineSubSubSectionfalse\@numberSubSubSectiontrue
	\@boldheadtrue\@fixedHeadSzfalse}
% -----------------------------------------------------------------------------
% Declare font options
\DeclareOption{Helvetica}{\renewcommand{\rmdefault}{phv}}
\DeclareOption{AvantGarde}{\renewcommand{\rmdefault}{pag}}
%
% -----------------------------------------------------------------------------
% Now we load the report class with certain options by default.  It
% will process any other options specified.  Then, we process the
% options for this class file.  Lastly, we load in the font option
% file for the given font size.
%
\ProcessOptions
\LoadClass[a4paper,twoside,onecolumn]{report}
%
\RequirePackage{indentfirst}
\RequirePackage{ifthen}
%\RequirePackage[ULforem,normalbf]{ulem}
\RequirePackage{graphicx}
\RequirePackage{epsf}
\RequirePackage[absolute]{textpos}
\RequirePackage{helvet}
%
% -----------------------------------------------------------------------------
% SPECIAL INTERNAL COMMANDS. (Needed by later routines)
%
% Create a command to be used for default arguments.
\newcommand*{\th@DefArg}{aWfdRvV11f66GL00K5}

% Now create a command to check for the default argument, and call the
% command [in the first arg] appropriately.  Used for section heading
% commands.

\newcommand{\th@ExecChkOptArg}[3]{
\ifthenelse{\equal{\th@DefArg}{#2}}{
	#1{#3}
	}{
	#1[#2]{#3}}
}
\newcommand{\th@StoreIn}[2]{ \gdef#1{#2} }
%
% -----------------------------------------------------------------------------
\newenvironment{setfont}[1]{\fontsize{#1}{#1}\selectfont}{}
%
% -----------------------------------------------------------------------------
% TITLEPAGE COMMANDS
%
% DEFINE COMMAND FOR: univlogo -> UNIVERISTY LOGO
\newcommand*{\univlogo}[3]{\th@StoreIn{\th@univlogo}{#3} \th@StoreIn{\th@univlogopsx}{#1} \th@StoreIn{\th@univlogopsy}{#2}}
\newcommand*{\th@univlogo}{
	\ClassError{istulthesis}{
		You must specify the university image logo}
	{Use the \protect\univlogo{} command in the preamble.}
}
% position coordinates of the university image logo
\newcommand*{\th@univlogopsx}{2}
\newcommand*{\th@univlogopsy}{2}
%
% DEFINE COMMAND FOR: thesislogo -> THESIS IMAGE
\newcommand*{\thesislogo}[3]{\th@StoreIn{\th@thesislogo}{#3} \th@StoreIn{\th@thesislogopsx}{#1} \th@StoreIn{\th@thesislogopsy}{#2}}
\newcommand*{\th@thesislogo}{\relax}
\newcommand*{\th@thesislogopsx}{2}
\newcommand*{\th@thesislogopsy}{2}
%
% -----------------------------------------------------------------------------
% DEFINE COMMAND FOR: degree -> DEGREE the AUTHOR is applying
\newcommand*{\degree}[1]{\th@StoreIn{\th@degree}{#1}}
\newcommand*{\th@degree}{ % command is required
	\ClassError{istulthesis}{
		You must specify the degree you are applying}
	{Use the \protect\degree{} command in the preamble.}
}
%
% DEFINE COMMAND FOR: supervisor -> NAME OF THESIS SCIENTIFIC SUPERVISOR
\newcommand*{\supervisor}[1]{\th@StoreIn{\th@supervisor}{#1}}
\newcommand*{\th@supervisor}{ % command is required
	\ClassError{istulthesis}{
		You must specify the scientific supervisor}
	{Use the \protect\supervisor{} command in the preamble.}
}
%
% DEFINE COMMAND FOR: othersupervisor -> NAME OF THESIS SCIENTIFIC CO-SUPERVISOR
\newcommand*{\othersupervisor}[1]{\th@StoreIn{\th@othersupervisor}{#1}}
\newcommand*{\th@othersupervisor}{\relax} % command is optional
%
% DEFINE COMMAND FOR: Examination Committee
\newcommand*{\chairperson}[1]{\th@StoreIn{\th@chairperson}{#1}}
\newcommand*{\th@chairperson}{\relax} % command is optional
% Vogal 1
\newcommand*{\vogalone}[1]{\th@StoreIn{\th@vogalone}{#1}}
\newcommand*{\th@vogalone}{\relax} % command is optional
% Vogal 2
\newcommand*{\vogaltwo}[1]{\th@StoreIn{\th@vogaltwo}{#1}}
\newcommand*{\th@vogaltwo}{\relax} % command is optional
% Vogal 3
\newcommand*{\vogalthree}[1]{\th@StoreIn{\th@vogalthree}{#1}}
\newcommand*{\th@vogalthree}{\relax} % command is optional
%
% DEFINE COMMAND FOR: finalthesis --> value 'true' prints
% the full commiteee list
\newcommand*{\finalthesis}[1]{\th@StoreIn{\th@finalthesis}{#1}}
\newcommand*{\th@finalthesis}{true}
%
% DEFINE COMMAND FOR: Thesis type, depending on language
\newcommand{\thesistypelang}{\IfLanguageName{english}
		{Thesis to obtain the Master of Science Degree in}
		{Disserta\c{c}\~ao para obten\c{c}\~ao do Grau de Mestre em}
        }
%
% DEFINE COMMAND FOR: Copyright Declaration, depending on language
\newcommand{\tlangDeclcopyright}{\IfLanguageName{english}
        {Declaration}
        {Declara\c{c}\~ao}}
%
% DEFINE COMMAND FOR: Copyright Declaration Text, depending on language
\newcommand{\thesisCopyrightlang}{\IfLanguageName{english}
		{\begin{quote}
            I declare that this document is an original work of my own authorship 
            and that it fulfills all the requirements of the Code of Conduct and 
            Good Practices of the Universidade de Lisboa.
        \end{quote}}
		{\begin{quote}
            Declaro que o presente documento \'e um trabalho original da minha autoria
            e que cumpre todos os requisitos do C\'odigo de Conduta e Boas Pr\'aticas 
            da Universidade de Lisboa.
        \end{quote}}
        }
%
% DEFINE COMMAND FOR: subtitle (optional)
\newcommand*{\subtitle}[1]{\th@StoreIn{\th@subtitle}{#1}}
\newcommand*{\th@subtitle}{\relax} % command is optional
%
% DEFINE COMMAND FOR: Supervisors prefix depending on language
\newcommand{\tlangSupervisors}{\IfLanguageName{english}{Supervisors: }{Orientadores: }}
%
% DEFINE COMMAND FOR: the only Supervisor prefix depending on language
\newcommand{\tlangComSupervisor}{\IfLanguageName{english}{Supervisor: }{Orientador: }}
%
% DEFINE COMMAND FOR: Examination Committee, depending on language
\newcommand{\tlangCommittee}{\IfLanguageName{english}{Examination Committee}{J\'uri}}
%
% DEFINE COMMAND FOR: Chairperson prefix depending on language
\newcommand{\tlangChairperson}{\IfLanguageName{english}{Chairperson: }{Presidente: }}
%
% DEFINE COMMAND FOR: Committee members prefix depending on language
\newcommand{\tlangMembers}{\IfLanguageName{english}{Members of the Committee: }{Vogais: }}

% DEFINE COMMAND FOR: Single Committee member prefix depending on language
\newcommand{\tlangMember}{\IfLanguageName{english}{Member of the Committee: }{Vogal: }}
%
% DEFINE COMMAND FOR: Acknowledgwments depending on language
\newcommand{\tlangAcknowledgments}{\IfLanguageName{english}{Acknowledgments}{Agradecimentos}}
%
% DEFINE COMMAND FOR: Acronyms listing Title depending on language
\newcommand{\tlangAcronyms}{\IfLanguageName{english}{Acronyms}{Acr\'onimos}}
%
% DEFINE COMMAND FOR: List of Symbols depending on language
\newcommand{\listofsymbolsname}{\IfLanguageName{english}{List of Symbols}{Lista de S\'imbolos}}
%
% #############################################################################
% CREATING THE TITLE PAGE (FRONTCOVER)
% #############################################################################
%
\setlength{\headheight}{14.5pt}
\setlength\oddsidemargin{2.5 cm}
\setlength\evensidemargin{2.5 cm}
%

% #############################################################################
% DEFINE the COPYRIGHT DECLARATION PAGE
% #############################################################################
%
\newenvironment{declcopyright}{\vspace*{3cm}
    {\parindent \z@ \raggedright \normalfont
        \hspace*{0\p@}\huge\bfseries \tlangDeclcopyright
        \par\nobreak
        \vspace*{1cm}
    }
}{
	\clearpage
	\thispagestyle{empty}
	\cleardoublepage
}

% #############################################################################
% REDEFINE the ABSTRACT and ACKNOWLEDGEMENTS
% #############################################################################
%
% English version
\renewenvironment{abstract}{
	\pdfbookmark{Abstract}{Abstract}
	\chapter*{Abstract}}
	
\newenvironment{keywords}{\vspace*{3cm}
	{\parindent \z@ \raggedright \normalfont
	\hspace*{0\p@}\huge\bfseries Keywords
	\par\nobreak
	\vspace*{1cm}
	}
}

% Portuguese version
\newenvironment{resumo}{
	\pdfbookmark{Resumo}{Resumo}
	\chapter*{Resumo}}

\newenvironment{palavraschave}{\vspace*{3cm}
    {\parindent \z@ \raggedright \normalfont
        \hspace*{0\p@}\huge\bfseries Palavras Chave
        \par\nobreak
        \vspace*{1cm}
    }
}


% The Acknowledgments
\newenvironment{acknowledgments}{\vspace*{3cm}
    {\parindent \z@ \raggedright \normalfont
        \hspace*{0\p@}\huge\bfseries \tlangAcknowledgments
        \par\nobreak
        \vspace*{1cm}
    }
}{
	\clearpage
	\thispagestyle{empty}
	\cleardoublepage
}

\newcommand{\tblc}[1]{\makebox[\linewidth][c]{#1}}
%
\let\@afterindentfalse\@afterindenttrue
\@afterindenttrue
%
\endinput
%%
%% End of file `istulthesis.cls'.