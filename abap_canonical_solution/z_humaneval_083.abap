CLASS z_humaneval_083 DEFINITION PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS:
      check
        IMPORTING iv        TYPE i
        RETURNING VALUE(rv) TYPE i.
ENDCLASS.



CLASS z_humaneval_083 IMPLEMENTATION.
  METHOD check.
    DATA: start_with_1    TYPE i,
          end_with_1      TYPE i,
          start_and_end_1 TYPE i.

    IF iv = 1.
      rv = 1.
    ELSE.
      start_with_1 = 10 ** ( iv - 1 ).
      end_with_1 = 9 * ( 10 ** ( iv - 2 ) ).
      start_and_end_1 = 10 ** ( iv - 2 ).
      rv = start_with_1 + end_with_1 - start_and_end_1.
    ENDIF.
  ENDMETHOD.
ENDCLASS.