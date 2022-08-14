package util
	
import "golang.org/x/xerrors"

func CreateErr(msg string) error {
	return xerrors.New(msg)
}

func WrapErr(msg string, err error) error {
	return xerrors.Errorf("%s: %w", msg, err)
}
