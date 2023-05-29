import sys, getopt

from server import initialize_mesa_server

options = ""
long_options = ["force-read", "debug"]

def main(argv):
    # Default values
    force_read = False
    debug_mode = False

    # Argument reading
    arguments, _ = getopt.getopt(argv, options, long_options)
    
    for currentArgument, _ in arguments:
        if currentArgument == '--force-read':
            force_read = True
            print("--- Forcing CSV data read ---") if debug_mode else 0
        if currentArgument == '--debug':
            debug_mode = True
            print("--- Debug mode ---") if debug_mode else 0
 
    initialize_mesa_server(force_read, debug_mode)
    
if __name__ == "__main__":
    main(sys.argv[1:])