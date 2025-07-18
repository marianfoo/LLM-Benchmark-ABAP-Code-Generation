CLASS z_humaneval_044 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: change_base
      IMPORTING
        VALUE(x)      TYPE i
        VALUE(base)   TYPE i
      RETURNING
        VALUE(result) TYPE string.

  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_044 IMPLEMENTATION.
  METHOD change_base.
    DATA: temp_x        TYPE i,
          remainder     TYPE i,
          remainder_str TYPE string.

    temp_x = x.

    WHILE temp_x > 0.
      remainder = temp_x MOD base.
      remainder_str = |{ remainder }|.
      result = |{ remainder_str }{ result }|.
      temp_x = temp_x DIV base.
    ENDWHILE.
  ENDMETHOD.
ENDCLASS.