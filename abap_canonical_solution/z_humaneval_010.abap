CLASS z_humaneval_010 DEFINITION
PUBLIC
FINAL
CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS make_palindrome
      IMPORTING
        VALUE(string) TYPE string
      RETURNING
        VALUE(result) TYPE string.

  PRIVATE SECTION.
    CLASS-METHODS is_palindrome
      IMPORTING
        VALUE(string) TYPE string
      RETURNING
        VALUE(is_pal) TYPE abap_bool.

ENDCLASS.


CLASS z_humaneval_010 IMPLEMENTATION.

  METHOD make_palindrome.
    IF string IS INITIAL.
      result = ''.
      RETURN.
    ENDIF.

    DATA(beginning_of_suffix) = 0.

    WHILE NOT is_palindrome( substring( val = string off = beginning_of_suffix ) ).
      beginning_of_suffix = beginning_of_suffix + 1.
    ENDWHILE.

    DATA(prefix) = substring( val = string off = 0 len = beginning_of_suffix ).
    result = string && reverse( prefix ).
  ENDMETHOD.

  METHOD is_palindrome.
    DATA(length) = strlen( string ).
    DATA(flag) = abap_true.

    DO length / 2 TIMES.
      DATA(left_char) = substring( val = string off = sy-index - 1 len = 1 ).
      DATA(right_char) = substring( val = string off = length - sy-index len = 1 ).
      IF left_char <> right_char.
        flag = abap_false.
        EXIT.
      ENDIF.
    ENDDO.

    is_pal = flag.
  ENDMETHOD.

ENDCLASS.