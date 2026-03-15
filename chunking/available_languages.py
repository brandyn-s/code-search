"""Language initialization for tree-sitter based chunkers.

This module handles importing and registering available languages
for the tree-sitter based code chunking system.
"""

import logging
from tree_sitter import Language

logger = logging.getLogger(__name__)

def get_availiable_language():
    """
    Return a map {language: language_obj}
    """
    # Try to import language bindings
    res = {}

    try:
        import tree_sitter_python as tspython
        res['python'] = Language(tspython.language())
    except ImportError:
        logger.debug("tree-sitter-python not installed")

    try:
        import tree_sitter_javascript as tsjavascript
        res['javascript'] = Language(tsjavascript.language())
        # JavaScript also supports JSX
        res['jsx'] = res['javascript']
    except ImportError:
        logger.debug("tree-sitter-javascript not installed for JSX")

    try:
        import tree_sitter_typescript as tstypescript
        # TypeScript has two grammars: typescript and tsx
        res['typescript'] = Language(tstypescript.language_typescript())
        res['tsx'] = Language(tstypescript.language_tsx())
    except ImportError:
        logger.debug("tree-sitter-typescript not installed")

    try:
        import tree_sitter_svelte as tssvelte
        res['svelte'] = Language(tssvelte.language())
    except ImportError:
        logger.debug("tree-sitter-svelte not installed")

    try:
        import tree_sitter_go as tsgo
        res['go'] = Language(tsgo.language())
    except ImportError:
        logger.debug("tree-sitter-go not installed")

    try:
        import tree_sitter_rust as tsrust
        res['rust'] = Language(tsrust.language())
    except ImportError:
        logger.debug("tree-sitter-rust not installed")

    try:
        import tree_sitter_java as tsjava
        res['java'] = Language(tsjava.language())
    except ImportError:
        logger.debug("tree-sitter-java not installed")

    try:
        import tree_sitter_c as tsc
        res['c'] = Language(tsc.language())
    except ImportError:
        logger.debug("tree-sitter-c not installed")

    try:
        import tree_sitter_cpp as tscpp
        res['cpp'] = Language(tscpp.language())
    except ImportError:
        logger.debug("tree-sitter-cpp not installed")

    try:
        import tree_sitter_c_sharp as tscsharp
        res['csharp'] = Language(tscsharp.language())
    except ImportError:
        logger.debug("tree-sitter-c-sharp not installed")

    try:
        import tree_sitter_markdown as tsmarkdown
        res['markdown'] = Language(tsmarkdown.language())
    except ImportError:
        logger.debug("tree-sitter-markdown not installed")

    try:
        import tree_sitter_nix as tsnix
        res['nix'] = Language(tsnix.language())
    except ImportError:
        logger.debug("tree-sitter-nix not installed")

    return res