CLASS ltcl_test_can_arrange DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS test_case_1 FOR TESTING.
    METHODS test_case_2 FOR TESTING.
    METHODS test_case_3 FOR TESTING.
    METHODS test_case_4 FOR TESTING.
    METHODS test_case_5 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_can_arrange IMPLEMENTATION.

  METHOD test_case_1.
    DATA(result) = z_humaneval_135=>can_arrange( VALUE int4_table( ( 1 ) ( 2 ) ( 4 ) ( 3 ) ( 5 ) ) ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 3 ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA(result) = z_humaneval_135=>can_arrange( VALUE int4_table( ( 1 ) ( 2 ) ( 4 ) ( 5 ) ) ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -1 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA(result) = z_humaneval_135=>can_arrange( VALUE int4_table( ( 1 ) ( 4 ) ( 2 ) ( 5 ) ( 6 ) ( 7 ) ( 8 ) ( 9 ) ( 10 ) ) ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 2 ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA(result) = z_humaneval_135=>can_arrange( VALUE int4_table( ( 4 ) ( 8 ) ( 5 ) ( 7 ) ( 3 ) ) ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 4 ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA(result) = z_humaneval_135=>can_arrange( VALUE int4_table( ) ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -1 ).
  ENDMETHOD.

ENDCLASS.