#!/usr/bin/env perl
# No strict 'vars': each block is a separate eval; bare $x = 1 should persist like
# perl -e one-liners. refs/subs stay strict for safety.
use strict qw(refs subs);
use warnings;
$| = 1;

# Persistent Perl REPL for Open Interpreter execute blocks.
# Blocks end with __OI_END__ on its own line (injected by perl.py preprocess_code).

while (1) {
    my @lines;
    while (my $line = <STDIN>) {
        chomp $line;
        if ($line eq "__OI_END__") {
            last;
        }
        push @lines, $line . "\n";
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
