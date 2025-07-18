CLASS z_humaneval_160 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: do_algebra
      IMPORTING
        operator      TYPE string_table
        operand       TYPE int4_table
      RETURNING
        VALUE(result) TYPE i.

  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_160 IMPLEMENTATION.

  METHOD do_algebra.
    DATA: values      TYPE int4_table,
          operators   TYPE string_table,
          temp_result TYPE f,
          i           TYPE i,
          j           TYPE i.

    values = operand.
    operators = operator.

    DO 2 TIMES.
      CASE sy-index.
        WHEN 1.
          i = 1.
          WHILE i <= lines( operators ).
            IF operators[ i ] = '**'.
              temp_result = values[ i ] ** values[ i + 1 ].
              values[ i ] = temp_result.
              DELETE values INDEX i + 1.
              DELETE operators INDEX i.
            ELSE.
              i = i + 1.
            ENDIF.
          ENDWHILE.

        WHEN 2.
          i = 1.
          WHILE i <= lines( operators ).
            IF operators[ i ] = '*' OR operators[ i ] = '//'.
              IF operators[ i ] = '*'.
                temp_result = values[ i ] * values[ i + 1 ].
              ELSE.
                temp_result = values[ i ] DIV values[ i + 1 ].
              ENDIF.
              values[ i ] = temp_result.
              DELETE values INDEX i + 1.
              DELETE operators INDEX i.
            ELSE.
              i = i + 1.
            ENDIF.
          ENDWHILE.
      ENDCASE.
    ENDDO.

    i = 1.
    WHILE i <= lines( operators ).
      IF operators[ i ] = '+'.
        values[ i ] = values[ i ] + values[ i + 1 ].
      ELSEIF operators[ i ] = '-'.
        values[ i ] = values[ i ] - values[ i + 1 ].
      ENDIF.
      DELETE values INDEX i + 1.
      DELETE operators INDEX i.
    ENDWHILE.

    result = values[ 1 ].

  ENDMETHOD.

ENDCLASS.