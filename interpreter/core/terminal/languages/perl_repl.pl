#!/usr/bin/env perl
use strict;
use warnings;
$| = 1;

# Persistent Perl REPL for Open Interpreter execute blocks.
# Blocks end with __OI_END__ on its own line (injected by perl.py preprocess_code).

while (1) {
    my @lines;
    while (my $line = <STDIN>) {
        if ($line eq "__OI_END__\n") {
            last;
        }
        push @lines, $line;
    }
    last unless @lines;

    my $code = join "", @lines;
    eval $code;
    if ($@) {
        chomp $@;
        print "##execution_error##\n$@\n";
    }
    print "##end_of_execution##\n";
}
