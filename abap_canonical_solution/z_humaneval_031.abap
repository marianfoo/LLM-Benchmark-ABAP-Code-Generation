CLASS z_humaneval_031 DEFINITION
  PUBLIC
  FINAL
  ABSTRACT.

  PUBLIC SECTION.
    CLASS-METHODS is_prime
      IMPORTING
        VALUE(n)      TYPE i
      RETURNING
        VALUE(result) TYPE abap_bool.
ENDCLASS.

CLASS z_humaneval_031 IMPLEMENTATION.

  METHOD is_prime.
    IF n < 2.
      result = abap_false.
      RETURN.
    ENDIF.

    DO n - 2 TIMES.
      IF n MOD ( sy-index + 1 ) = 0.
        result = abap_false.
        RETURN.
      ENDIF.
    ENDDO.

    result = abap_true.
  ENDMETHOD.

ENDCLASS.