package util
	
import "golang.org/x/xerrors"

func WrapErr(msg string, err error) error {
	return xerrors.Errorf("%s: %w", msg, err)
}
