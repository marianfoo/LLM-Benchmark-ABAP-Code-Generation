CLASS ltcl_test_move_one_ball DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_move_one_ball IMPLEMENTATION.

  METHOD test_case_1.
    DATA: arr    TYPE int4_table,
          result TYPE abap_bool.

    arr = VALUE #( ( 3 ) ( 4 ) ( 5 ) ( 1 ) ( 2 ) ).
    result = z_humaneval_109=>move_one_ball( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: arr    TYPE int4_table,
          result TYPE abap_bool.

    arr = VALUE #( ( 3 ) ( 5 ) ( 10 ) ( 1 ) ( 2 ) ).
    result = z_humaneval_109=>move_one_ball( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: arr    TYPE int4_table,
          result TYPE abap_bool.

    arr = VALUE #( ( 4 ) ( 3 ) ( 1 ) ( 2 ) ).
    result = z_humaneval_109=>move_one_ball( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: arr    TYPE int4_table,
          result TYPE abap_bool.

    arr = VALUE #( ( 3 ) ( 5 ) ( 4 ) ( 1 ) ( 2 ) ).
    result = z_humaneval_109=>move_one_ball( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: arr    TYPE int4_table,
          result TYPE abap_bool.

    result = z_humaneval_109=>move_one_ball( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_true ).
  ENDMETHOD.

ENDCLASS.