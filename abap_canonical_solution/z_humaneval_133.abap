CLASS z_humaneval_133 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF float_type,
             value TYPE f,
           END OF float_type,
           float_table TYPE TABLE OF float_type.

    CLASS-METHODS: sum_squares
      IMPORTING lst           TYPE float_table
      RETURNING VALUE(result) TYPE i.

  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_133 IMPLEMENTATION.

  METHOD sum_squares.
    DATA: squared     TYPE i VALUE 0,
          ceiling_val TYPE i.

    LOOP AT lst INTO DATA(element).
      ceiling_val = ceil( element-value ).
      squared = squared + ( ceiling_val * ceiling_val ).
    ENDLOOP.

    result = squared.
  ENDMETHOD.

ENDCLASS.