CLASS z_humaneval_002 DEFINITION
  PUBLIC
  FINAL
  ABSTRACT.

  PUBLIC SECTION.
    CLASS-METHODS truncate_number
      IMPORTING
        VALUE(number)       TYPE f
      RETURNING
        VALUE(decimal_part) TYPE f.

ENDCLASS.

CLASS z_humaneval_002 IMPLEMENTATION.

  METHOD truncate_number.
    decimal_part = number - floor( number ).
  ENDMETHOD.

ENDCLASS.
